#!/usr/bin/env python3
"""
Automated Documentation Pipeline

Build an automated pipeline for continuous network documentation generation.

From: AI for Networking Engineers - Volume 2, Chapter 13
Author: Eduard Dulharu (Ed Harmoosh)

This module provides:
- Batch configuration processing
- Index page generation
- Git version control integration
- Scheduled documentation updates
- Change detection and incremental updates

Usage:
    # Command line
    python documentation_pipeline.py --generate-now --config-dir ./configs

    # Programmatic
    from documentation_pipeline import DocumentationPipeline

    pipeline = DocumentationPipeline(config_dir="./configs", output_dir="./docs")
    pipeline.generate_all_documentation()
"""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse
import os
import hashlib

# Local import - assumes doc_generator.py is in same directory
try:
    from doc_generator import ConfigDocumentationGenerator
except ImportError:
    # Handle case where running from different directory
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from doc_generator import ConfigDocumentationGenerator


class DocumentationPipeline:
    """
    Automated pipeline for network documentation.

    Processes multiple device configurations, generates documentation,
    and optionally commits changes to a Git repository.

    Features:
    - Batch processing of config files
    - Index page generation with links to all docs
    - Git integration for version control
    - Scheduled daily updates
    - Change detection (via config hashing)

    Attributes:
        generator: ConfigDocumentationGenerator instance
        config_dir: Path to directory containing .cfg files
        output_dir: Path to output documentation directory
        git_repo: Optional Git repository for versioning

    Example:
        >>> pipeline = DocumentationPipeline(
        ...     config_dir="./configs",
        ...     output_dir="./docs"
        ... )
        >>> pipeline.generate_all_documentation()
    """

    def __init__(
        self,
        api_key: str = None,
        config_dir: str = "./configs",
        output_dir: str = "./docs",
        git_repo: str = None
    ):
        """
        Initialize the documentation pipeline.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            config_dir: Directory containing device configuration files (.cfg)
            output_dir: Directory for generated documentation
            git_repo: Optional path to Git repository for version control
        """
        self.generator = ConfigDocumentationGenerator(api_key)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.git_repo = git_repo
        self.repo = None

        # Initialize Git if configured
        if git_repo:
            try:
                import git
                self.repo = git.Repo(git_repo)
                print(f"✓ Git repository initialized: {git_repo}")
            except ImportError:
                print("⚠ Warning: GitPython not installed. Git versioning disabled.")
                print("  Install with: pip install gitpython")
            except Exception as e:
                print(f"⚠ Warning: Could not initialize Git repo: {e}")

        # Hash file for change detection
        self.hash_file = self.output_dir / ".config_hashes.json"

    def _get_config_hash(self, config: str) -> str:
        """Calculate MD5 hash of config for change detection."""
        return hashlib.md5(config.encode()).hexdigest()

    def _load_hashes(self) -> Dict[str, str]:
        """Load previous config hashes from file."""
        if self.hash_file.exists():
            import json
            with open(self.hash_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_hashes(self, hashes: Dict[str, str]):
        """Save config hashes to file."""
        import json
        with open(self.hash_file, 'w') as f:
            json.dump(hashes, f, indent=2)

    def fetch_configs(self) -> Dict[str, str]:
        """
        Fetch current configs from config directory.

        In production, this could be extended to:
        - Pull configs via Netmiko/NAPALM
        - Clone from a config Git repository
        - Query a network source of truth

        Returns:
            Dictionary mapping hostname to configuration text
        """
        configs = {}

        if not self.config_dir.exists():
            print(f"⚠ Config directory does not exist: {self.config_dir}")
            return configs

        # Support multiple file extensions
        for ext in ['*.cfg', '*.conf', '*.txt']:
            for config_file in self.config_dir.glob(ext):
                hostname = config_file.stem
                with open(config_file, 'r') as f:
                    configs[hostname] = f.read()

        return configs

    def generate_all_documentation(self, force: bool = False):
        """
        Generate documentation for all devices.

        Processes all configuration files in the config directory,
        generates documentation, creates an index page, and optionally
        commits to Git.

        Args:
            force: If True, regenerate all docs even if config unchanged
        """
        print(f"\n{'='*60}")
        print(f"Documentation Generation Started: {datetime.now()}")
        print(f"{'='*60}\n")

        configs = self.fetch_configs()
        print(f"Found {len(configs)} device configs in {self.config_dir}")

        if not configs:
            print("No configuration files found. Nothing to generate.")
            return

        # Load previous hashes for change detection
        previous_hashes = self._load_hashes()
        current_hashes = {}

        generated_files = []
        skipped_count = 0

        for hostname, config in sorted(configs.items()):
            try:
                # Check if config has changed
                current_hash = self._get_config_hash(config)
                current_hashes[hostname] = current_hash

                if not force and hostname in previous_hashes:
                    if previous_hashes[hostname] == current_hash:
                        print(f"  → {hostname} (unchanged, skipping)")
                        # Still add to generated files for index
                        output_file = self.output_dir / f"{hostname}.md"
                        if output_file.exists():
                            generated_files.append(output_file)
                        skipped_count += 1
                        continue

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

        # Save current hashes
        self._save_hashes(current_hashes)

        # Generate index
        self.generate_index(generated_files)

        # Commit to Git if configured
        if self.repo:
            self.commit_changes()

        print(f"\n{'='*60}")
        print(f"Documentation Generation Complete")
        print(f"{'='*60}")
        print(f"  Total configs: {len(configs)}")
        print(f"  Generated: {len(generated_files) - skipped_count}")
        print(f"  Skipped (unchanged): {skipped_count}")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

    def generate_index(self, doc_files: List[Path]):
        """
        Generate index page linking to all device documentation.

        Creates a README.md in the output directory with links to
        all generated documentation files.

        Args:
            doc_files: List of paths to generated documentation files
        """
        # Filter to only existing files
        existing_files = [f for f in doc_files if f.exists()]

        index_content = f"""# Network Documentation Index

**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Devices**: {len(existing_files)}

---

## Quick Links

| Device | Last Updated |
|--------|--------------|
"""

        for doc_file in sorted(existing_files):
            hostname = doc_file.stem
            # Get file modification time
            mtime = datetime.fromtimestamp(doc_file.stat().st_mtime)
            mtime_str = mtime.strftime('%Y-%m-%d %H:%M')
            index_content += f"| [{hostname}]({doc_file.name}) | {mtime_str} |\n"

        index_content += f"""

---

## About This Documentation

This documentation is **automatically generated** from device configurations using AI.

### Generation Details
- **Generator**: AI-powered documentation pipeline (Chapter 13)
- **Update Frequency**: Daily at 2:00 AM (or on config change)
- **Source**: Device running configurations

### Manual Update

To regenerate documentation manually:

```bash
python documentation_pipeline.py --generate-now --config-dir ./configs
```

### Forcing Full Regeneration

To regenerate all docs even if configs haven't changed:

```bash
python documentation_pipeline.py --generate-now --force --config-dir ./configs
```

---

## Documentation Standards

All device documentation follows a consistent format:

1. **Overview** - Device role, management IP, key features
2. **Interfaces** - Table of all interfaces with IPs and descriptions
3. **Routing** - Protocols, neighbors, policies
4. **Security** - ACLs, authentication, management access

---

*Generated by AI for Networking Engineers - Volume 2, Chapter 13*
"""

        index_file = self.output_dir / "README.md"
        with open(index_file, 'w') as f:
            f.write(index_content)

        print(f"  ✓ Generated index: {index_file}")

    def commit_changes(self):
        """
        Commit documentation updates to Git.

        Stages all changes in the output directory and creates
        a commit with a timestamp. Does not push by default.
        """
        if not self.repo:
            return

        try:
            # Stage all documentation changes
            self.repo.git.add(str(self.output_dir))

            if self.repo.is_dirty():
                commit_message = f"Auto-update network documentation - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                self.repo.index.commit(commit_message)
                print(f"  ✓ Changes committed to Git")

                # Optional: Push to remote
                # Uncomment the following lines to auto-push:
                # origin = self.repo.remote(name='origin')
                # origin.push()
                # print(f"  ✓ Pushed to remote")
            else:
                print(f"  → No changes to commit")

        except Exception as e:
            print(f"  ✗ Git commit failed: {e}")

    def schedule_daily_updates(self):
        """
        Schedule automatic daily documentation updates.

        Runs the documentation generation daily at 2:00 AM.
        Uses the 'schedule' library for timing.
        """
        try:
            import schedule
            import time
        except ImportError:
            print("Error: 'schedule' package not installed.")
            print("Install with: pip install schedule")
            return

        # Run daily at 2 AM
        schedule.every().day.at("02:00").do(self.generate_all_documentation)

        print("=" * 60)
        print("Documentation Pipeline Scheduled")
        print("=" * 60)
        print(f"  Schedule: Daily at 2:00 AM")
        print(f"  Config directory: {self.config_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Git versioning: {'Enabled' if self.repo else 'Disabled'}")
        print("=" * 60)
        print("\nPress Ctrl+C to stop")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nScheduler stopped.")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Network Documentation Pipeline - Auto-generate docs from configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate documentation now
  python documentation_pipeline.py --generate-now

  # Generate with custom directories
  python documentation_pipeline.py --generate-now --config-dir ./configs --output-dir ./docs

  # Force regeneration of all docs
  python documentation_pipeline.py --generate-now --force

  # Schedule daily updates
  python documentation_pipeline.py --schedule

  # With Git versioning
  python documentation_pipeline.py --generate-now --git-repo /path/to/docs/repo
        """
    )

    parser.add_argument(
        "--generate-now",
        action="store_true",
        help="Generate documentation immediately"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run scheduled daily updates (2:00 AM)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if configs unchanged"
    )
    parser.add_argument(
        "--config-dir",
        default="./configs",
        help="Directory containing device configs (default: ./configs)"
    )
    parser.add_argument(
        "--output-dir",
        default="./docs",
        help="Output directory for documentation (default: ./docs)"
    )
    parser.add_argument(
        "--git-repo",
        default=None,
        help="Git repository path for versioning"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = DocumentationPipeline(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        git_repo=args.git_repo
    )

    if args.generate_now:
        pipeline.generate_all_documentation(force=args.force)

    elif args.schedule:
        pipeline.schedule_daily_updates()

    else:
        # Show help and demo
        print("=" * 60)
        print("Chapter 13: Network Documentation Basics")
        print("Automated Documentation Pipeline")
        print("=" * 60)
        print("\nUsage:")
        print("  --generate-now    Generate documentation immediately")
        print("  --schedule        Run scheduled daily updates")
        print("  --force           Force regeneration (with --generate-now)")
        print("  --config-dir DIR  Directory with .cfg files")
        print("  --output-dir DIR  Output directory")
        print("  --git-repo PATH   Git repo for versioning")
        print("\nExample:")
        print("  python documentation_pipeline.py --generate-now --config-dir ./configs")
        print("\nRun with --help for full options.")


if __name__ == "__main__":
    main()
