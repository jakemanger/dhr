# Running DHR on Smithsonian Hydra (SI/HCP)

The Smithsonian Institution High Performance Cluster provides an option for high performance computing at SI. This guide covers setting up and running DHR on Hydra.

Information on Hydra can be found [here](https://confluence.si.edu/display/HPC/Overview).

The request form is found under "Hydra Policies", and the training is found under "Hydra Training". It may also be helpful to read through the reference pages as needed.

## Setting up Hydra

See [here](https://confluence.si.edu/pages/viewpage.action?pageId=163152227) for more information and use of command line for transfer.

### Initial Setup

1. **Connect via SFTP**: Using any software that supports secure file transfer (such as FileZilla), connect to:
   ```
   Host: sftp://hydra-login01.si.edu
   Username: yourusername
   Password: yourpassword
   ```

2. **Navigate to workspace**: Change the remote site from `/home/yourusername` to your workspace location
   ```
   Example: /scratch/public/genomics/yourusername
   ```

3. **Upload repository**: Clone this repository from GitHub and upload it to Hydra

4. **SSH connection**: Connect to Hydra through command line interface
   ```bash
   ssh yourusername@hydra-login01.si.edu
   ```

5. **Navigate to workspace**:
   ```bash
   cd /scratch/public/genomics/yourusername
   ```
   (Replace with your actual directory path)

### Environment Setup

1. **Load Python and create virtual environment**:
   ```bash
   module load tools/python/3.9
   python -m venv venv
   ```
   See [here](https://towardsdatascience.com/getting-started-with-python-virtual-environments-252a6bd2240) for more information on virtual environments.

2. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up directories**: Replace the symbolic dataset, logs, and output folders with new folders of the same names. Add a folder for scans. Because the program is being run on the cluster, any data you need has to be stored on Hydra. You can create other folders as needed.

## Running Through Job Files

Running programs through Hydra is similar to running them normally, but requires submitting commands through `.job` files. The job file will take care of loading the necessary tools and activating the virtual environment.

### Available Job Templates

Generating a dataset, training a model, and running inference should all be done through `.job` files. Templates are provided:
- `dataset_gen.job` - For dataset generation
- `train_template.job` - For model training
- `infer_template.job` - For inference

These should be copied and modified according to your needs.

### Job Management Commands

- **Submit job**: `qsub jobname.job`
- **Monitor jobs**: `qstat`
- **Check completed/failed job**: `qacct+ -j XXXXXXX` (where XXXXXXX is the job number)

**Note**: Running on a server prevents access to the napari miniviewer and similar GUI tools.

## Interactive Use

For running `.ipynb` files or interactive work, you can connect to Jupyter Lab on Hydra.

### Method 1: Direct Commands

While connected to Hydra, run:

```bash
qrsh -l gpu
module load tools/mamba
start-mamba
mamba activate dhr
module unload gcc
module load nvidia
jupyter lab --no-browser --ip=`hostname` --port=8888
```

### Method 2: Using Job File

```bash
qsub jupyter.job
cat jupyter.log
```

### Connecting to Jupyter

The output will include URLs like:

```
To access the server, open this file in a browser:
    file:///home/user/.local/share/....html
Or copy and paste one of these URLs:
    http://compute-XX-XX:8888/lab?token=1a2b3c4d5e6f7g8h9i10j11k12l13m14n15o16p17q18r
    http://127.0.0.1:8888/lab?token=1a2b3c4d5e6f7g8h9i10j11k12l13m14n15o16p17q18r
```

In a **new terminal** (without connecting to Hydra), create an SSH tunnel:

```bash
ssh -N -L 8888:compute-XX-XX:8888 yourusername@hydra-login01.si.edu
```

Replace:
- `XX-XX` with the node listed in the first terminal
- `yourusername` with your Hydra username

Then click the bottom link (featuring 127.0.0.1) to open Jupyter Lab on your computer.

## Tips for Hydra Usage

1. **Data Storage**: All data must be stored on Hydra since you're running on the cluster
2. **Job Files**: Always use job files for long-running processes
3. **Resource Allocation**: Be mindful of resource requests in your job files
4. **Monitoring**: Regularly check job status to ensure proper execution
5. **Workspace Management**: Keep your workspace organized and clean up temporary files

## Additional Resources

- [Hydra Documentation](https://confluence.si.edu/display/HPC/Overview)
- Hydra Policies (see main documentation)
- Hydra Training materials (see main documentation)
- Reference pages for specific tools and modules