#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to migrate SageMaker notebook instances to new platform identifier.

This script helps migrate notebook instances from deprecated platform identifiers
(like notebook-al2-v2) to the latest supported version (notebook-al2023-v1).

Usage:
    python migrate_notebooks.py [--profile PROFILE] <notebook-instance-name-1> [<notebook-instance-name-2> ...]
"""

import argparse
import sys
import boto3
from botocore.exceptions import ClientError


# Default SageMaker notebook platform identifier
# Update this when AWS releases newer platform versions
# See: https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-al2.html
DEFAULT_NOTEBOOK_PLATFORM = "notebook-al2023-v1"


def migrate_notebook(client, notebook_name, new_platform=None):
    """
    Migrate a SageMaker notebook instance to a new platform identifier.
    
    Args:
        client: boto3 SageMaker client
        notebook_name: Name of the notebook instance
        new_platform: Target platform identifier (default: from DEFAULT_NOTEBOOK_PLATFORM)
    """
    # Use default if not specified
    if new_platform is None:
        new_platform = DEFAULT_NOTEBOOK_PLATFORM
    
    print(f"\n{'='*60}")
    print(f"Processing notebook: {notebook_name}")
    print('='*60)
    
    try:
        # Get current notebook status
        response = client.describe_notebook_instance(
            NotebookInstanceName=notebook_name
        )
        current_status = response['NotebookInstanceStatus']
        current_platform = response.get('PlatformIdentifier', 'unknown')
        
        print(f"Current status: {current_status}")
        print(f"Current platform: {current_platform}")
        
        # Handle notebook state
        if current_status == 'InService':
            print(f"Stopping notebook instance...")
            client.stop_notebook_instance(NotebookInstanceName=notebook_name)
            
            # Wait for it to stop
            print("Waiting for notebook to stop...", end='', flush=True)
            waiter = client.get_waiter('notebook_instance_stopped')
            waiter.wait(
                NotebookInstanceName=notebook_name,
                WaiterConfig={'Delay': 10, 'MaxAttempts': 60}
            )
            print(" STOPPED")
        elif current_status == 'Stopped':
            print("Notebook is already stopped")
        elif current_status == 'Stopping':
            print("Notebook is currently stopping, waiting...")
            waiter = client.get_waiter('notebook_instance_stopped')
            waiter.wait(
                NotebookInstanceName=notebook_name,
                WaiterConfig={'Delay': 10, 'MaxAttempts': 60}
            )
            print(" STOPPED")
        else:
            print(f"ERROR: Notebook is in '{current_status}' state. Cannot proceed.")
            print("Please wait for the notebook to finish its current operation.")
            return False
        
        # Update the platform identifier
        print(f"Updating platform identifier to {new_platform}...")
        client.update_notebook_instance(
            NotebookInstanceName=notebook_name,
            PlatformIdentifier=new_platform
        )
        print(f"✓ Successfully updated to {new_platform}")
        
        # Optionally start the notebook
        response = input(f"Start notebook instance now? [y/N]: ")
        if response.lower() == 'y':
            print("Starting notebook instance...")
            client.start_notebook_instance(NotebookInstanceName=notebook_name)
            print(f"✓ Notebook {notebook_name} is starting")
        else:
            print(f"Notebook {notebook_name} remains stopped. Start it manually when ready.")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        print(f"ERROR: {error_code} - {error_msg}")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate SageMaker notebook instances to new platform identifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate using default AWS profile
  python migrate_notebooks.py my-notebook-1 my-notebook-2
  
  # Migrate using specific AWS profile
  python migrate_notebooks.py --profile production my-notebook-1
  
  # Migrate to specific platform version
  python migrate_notebooks.py --platform notebook-al2-v3 my-notebook-1
"""
    )
    parser.add_argument(
        'notebook_names',
        nargs='+',
        metavar='NOTEBOOK_NAME',
        help='Name(s) of notebook instance(s) to migrate'
    )
    parser.add_argument(
        '--profile',
        default=None,
        help='AWS profile to use (default: uses AWS_PROFILE env var or default profile)'
    )
    parser.add_argument(
        '--platform',
        default=None,
        help=f'Target platform identifier (default: {DEFAULT_NOTEBOOK_PLATFORM})'
    )
    
    args = parser.parse_args()
    
    print("SageMaker Notebook Migration Tool")
    print(f"Migrating {len(args.notebook_names)} notebook instance(s) to {args.platform or DEFAULT_NOTEBOOK_PLATFORM}")
    
    # Initialize boto3 client with optional profile
    try:
        if args.profile:
            print(f"Using AWS profile: {args.profile}")
            session = boto3.Session(profile_name=args.profile)
            client = session.client('sagemaker')
        else:
            client = boto3.client('sagemaker')
        print("✓ Connected to AWS SageMaker")
    except Exception as e:
        print(f"ERROR: Failed to connect to AWS: {e}")
        sys.exit(1)
    
    # Process each notebook
    results = {}
    for notebook_name in args.notebook_names:
        success = migrate_notebook(client, notebook_name, args.platform)
        results[notebook_name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("MIGRATION SUMMARY")
    print('='*60)
    for notebook_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{notebook_name}: {status}")
    
    successful = sum(1 for s in results.values() if s)
    print(f"\nTotal: {successful}/{len(args.notebook_names)} successful")


if __name__ == "__main__":
    main()
