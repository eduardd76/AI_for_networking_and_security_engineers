"""
Chapter 13: Network Documentation Basics
Automated Documentation Pipeline

Build an automated pipeline that generates and maintains network documentation.

How it works:
    1. Point this script at a directory full of .cfg files (device configs)
    2. It generates Markdown documentation for every device
    3. Optionally commits the docs to a Git repo for version history
    4. Can run on a schedule (daily at 2 AM) or as a one-shot

In production, you'd typically:
    - Store configs in Git (pulled by Oxidized, RANCID, or Ansible)
    - Run this pipeline in CI/CD (GitHub Actions, GitLab CI)
    - Have the generated docs auto-published to your wiki/Confluence

Prerequisites:
    pip install anthropic
    pip install gitpython   # optional, for Git versioning
    pip install schedule    # optional, for scheduled runs
    export ANTHROPIC_API_KEY="your-key-here"

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Import the generator (assumes it's in the same directory)
from doc_generator import ConfigDocumentationGenerator


class DocumentationPipeline:
    """Automated pipeline for network documentation."""

    def __init__(
        self,
        api_key: str = None,
        config_dir: str = "./configs",
        output_dir: str = "./docs",
        git_repo: str = None
    ):
        self.generator = ConfigDocumentationGenerator(api_key)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.git_repo = git_repo

        if git_repo:
            try:
                import git
                self.repo = git.Repo(git_repo)
            except ImportError:
                print("Warning: GitPython not installed. Git versioning disabled.")
                self.repo = None
        else:
            self.repo = None

    def fetch_configs(self) -> Dict[str, str]:
        """Fetch current configs from devices or config management system."""
        # In production: use Netmiko, NAPALM, or pull from Git

        configs = {}
        for config_file in self.config_dir.glob("*.cfg"):
            hostname = config_file.stem
            with open(config_file, 'r') as f:
                configs[hostname] = f.read()

        return configs

    def generate_all_documentation(self):
        """Generate documentation for all devices."""

        print(f"\n{'='*60}")
        print(f"Documentation Generation Started: {datetime.now()}")
        print(f"{'='*60}\n")

        configs = self.fetch_configs()
        print(f"Found {len(configs)} device configs")

        generated_files = []

        for hostname, config in configs.items():
            try:
                output_file = self.output_dir / f"{hostname}.md"

                self.generator.generate_complete_documentation(
                    config=config,
                    hostname=hostname,
                    output_file=str(output_file)
                )

                generated_files.append(output_file)
                print(f"  ✓ {hostname}")

            except Exception as e:
                print(f"  ✗ {hostname}: {e}")

        # Generate index
        self.generate_index(generated_files)

        # Commit to Git if configured
        if self.repo:
            self.commit_changes()

        print(f"\n{'='*60}")
        print(f"Documentation Generation Complete")
        print(f"Total files: {len(generated_files)}")
        print(f"{'='*60}\n")

    def generate_index(self, doc_files: List[Path]):
        """Generate index page linking to all device docs."""

        index_content = f"""# Network Documentation Index

**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Devices**: {len(doc_files)}

---

## Device Documentation

"""

        for doc_file in sorted(doc_files):
            hostname = doc_file.stem
            index_content += f"- [{hostname}]({doc_file.name})\n"

        index_content += """

---

## Documentation Standards

This documentation is automatically generated from device configurations.

**Update Frequency**: Daily at 2:00 AM
**Source**: Device running configurations
**Generator**: AI-powered documentation pipeline

To update manually: `python documentation_pipeline.py --generate-now`

---

*Auto-generated network documentation*
"""

        index_file = self.output_dir / "index.md"
        with open(index_file, 'w') as f:
            f.write(index_content)

        print(f"  ✓ Generated index: {index_file}")

    def commit_changes(self):
        """Commit documentation updates to Git."""

        try:
            self.repo.git.add(all=True)

            if self.repo.is_dirty():
                commit_message = f"Auto-update network documentation - {datetime.now()}"
                self.repo.index.commit(commit_message)

                # Push to remote (optional)
                # self.repo.remote(name='origin').push()

                print(f"  ✓ Changes committed to Git")
            else:
                print(f"  → No changes to commit")

        except Exception as e:
            print(f"  ✗ Git commit failed: {e}")

    def schedule_daily_updates(self):
        """Schedule automatic daily documentation updates."""
        
        try:
            import schedule
            import time
        except ImportError:
            print("Error: 'schedule' package not installed. Run: pip install schedule")
            return

        # Run daily at 2 AM
        schedule.every().day.at("02:00").do(self.generate_all_documentation)

        print("Documentation pipeline scheduled (daily at 2:00 AM)")
        print("Press Ctrl+C to stop")

        while True:
            schedule.run_pending()
            time.sleep(60)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network Documentation Pipeline")
    parser.add_argument("--generate-now", action="store_true",
                       help="Generate documentation immediately")
    parser.add_argument("--schedule", action="store_true",
                       help="Run scheduled updates")
    parser.add_argument("--config-dir", default="./configs",
                       help="Directory containing device configs")
    parser.add_argument("--output-dir", default="./docs",
                       help="Output directory for documentation")
    parser.add_argument("--git-repo", default=None,
                       help="Git repository path for versioning")

    args = parser.parse_args()

    pipeline = DocumentationPipeline(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        git_repo=args.git_repo
    )

    if args.generate_now:
        pipeline.generate_all_documentation()

    elif args.schedule:
        pipeline.schedule_daily_updates()

    else:
        print("Network Documentation Pipeline")
        print("=" * 40)
        print("\nUsage:")
        print("  --generate-now    Generate documentation immediately")
        print("  --schedule        Run scheduled daily updates")
        print("  --config-dir      Directory with .cfg files (default: ./configs)")
        print("  --output-dir      Output directory (default: ./docs)")
        print("  --git-repo        Git repo path for versioning")
        print("\nExample:")
        print("  python documentation_pipeline.py --generate-now --config-dir ./configs")
