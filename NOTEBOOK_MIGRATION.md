# SageMaker Notebook Platform Migration Guide

## Problem
You're getting this error when trying to start your SageMaker notebook instances:

```
ValidationException: Platform identifier (notebook-al2-v2) is not supported for this service. 
Please self migrate by updating the notebook settings
```

This happens because `notebook-al2-v2` (with JupyterLab 3) was deprecated as of June 30, 2025.

## Solutions

### Option 1: Using the ML2P CLI (New Command)

I've added an `update` command to ML2P. After pulling these changes:

```bash
# Stop, update, and start a notebook instance
ml2p notebook stop <notebook-name>

# Wait for it to stop (check with describe)
ml2p notebook describe <notebook-name>

# Update to the new platform
ml2p notebook update <notebook-name> --platform-identifier notebook-al2023-v1

# Start it again
ml2p notebook start <notebook-name>
```

**Example:**
```bash
ml2p notebook stop my-analysis-notebook
ml2p notebook update my-analysis-notebook
ml2p notebook start my-analysis-notebook
```

### Option 2: Using the Migration Script

I've created a standalone Python script that handles the entire process:

```bash
python migrate_notebooks.py <notebook-1> <notebook-2>
```

The script will:
1. Check the current status of each notebook
2. Stop the notebook if it's running
3. Update the platform identifier to `notebook-al2023-v1`
4. Ask if you want to start it immediately

**Example:**
```bash
python migrate_notebooks.py my-notebook-1 my-notebook-2
```

### Option 3: AWS CLI Commands

If you prefer using AWS CLI directly:

```bash
# For each notebook instance:
NOTEBOOK_NAME="your-notebook-name"

# Stop the notebook
aws sagemaker stop-notebook-instance --notebook-instance-name $NOTEBOOK_NAME

# Wait for it to stop
aws sagemaker wait notebook-instance-stopped --notebook-instance-name $NOTEBOOK_NAME

# Update the platform identifier
aws sagemaker update-notebook-instance \
  --notebook-instance-name $NOTEBOOK_NAME \
  --platform-identifier notebook-al2023-v1

# Start it again
aws sagemaker start-notebook-instance --notebook-instance-name $NOTEBOOK_NAME
```

### Option 4: Python/Boto3 Script

```python
import boto3

client = boto3.client('sagemaker')
notebook_name = 'your-notebook-name'

# Stop
client.stop_notebook_instance(NotebookInstanceName=notebook_name)
waiter = client.get_waiter('notebook_instance_stopped')
waiter.wait(NotebookInstanceName=notebook_name)

# Update
client.update_notebook_instance(
    NotebookInstanceName=notebook_name,
    PlatformIdentifier='notebook-al2023-v1'
)

# Start
client.start_notebook_instance(NotebookInstanceName=notebook_name)
```

## Important Notes

1. **Data Preservation**: Your notebook files and data are stored on an EBS volume that persists through the platform update. Your data will NOT be lost.

2. **Must Stop First**: You cannot update a running notebook instance - it must be in the "Stopped" state.

3. **Recommended Platform**: Use `notebook-al2023-v1` (Amazon Linux 2023 with JupyterLab 4) as it's the latest recommended version.

4. **For ML2P Projects**: If you manage notebooks via ml2p.yml, you can now add:
   ```yaml
   notebook:
     instance_type: "ml.t2.medium"
     volume_size: 8
     platform_identifier: "notebook-al2023-v1"  # Add this line
   ```

## Available Platform Identifiers

- `notebook-al2023-v1` - Amazon Linux 2023 + JupyterLab 4 (recommended)
- `notebook-al2-v3` - Amazon Linux 2 + JupyterLab 3 (if available in your region)

## What Changed in ML2P

1. **New CLI Command**: `ml2p notebook update` to update existing notebook instances
2. **Updated Notebook Creation**: `mk_notebook()` now defaults to `notebook-al2023-v1` for new notebooks
3. **Config Support**: You can specify `platform_identifier` in your ml2p.yml file

## Troubleshooting

**Q: Can't access my notebook to get my data?**  
A: Your data is safe on the EBS volume. Just update the platform identifier and you'll be able to access everything.

**Q: Update fails with "ResourceNotFound"?**  
A: Make sure you're using the correct notebook instance name. For ML2P projects, the full name is `{project-name}-{notebook-name}`.

**Q: Getting "ResourceInUse" error?**  
A: The notebook must be stopped first. Wait for it to fully stop before updating.
