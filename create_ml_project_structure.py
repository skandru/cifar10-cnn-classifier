#!/usr/bin/env python3
"""
CIFAR-10 CNN Project Structure Generator - Directory and Empty Files Only

This script creates the complete directory structure and empty files for the
CIFAR-10 CNN classifier project. You can then manually copy code into each file.

Usage:
    python Create_ML_Project_Structure.py [project_name]

Example:
    python Create_ML_Project_Structure.py my-cifar10-project
    python Create_ML_Project_Structure.py  # Uses default name 'cifar10-cnn-classifier'
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime


class ProjectStructureGenerator:
    """Generate ML project structure with empty files"""
    
    def __init__(self, project_name="cifar10-cnn-classifier"):
        self.project_name = project_name
        self.project_path = Path(project_name)
        self.created_files = []
        self.created_dirs = []
    
    def create_directory_structure(self):
        """Create all necessary directories"""
        directories = [
            "src",
            "notebooks", 
            "checkpoints",
            "results",
            "configs",
            "logs",
            "data",  # For dataset storage
            "tests",  # For unit tests
        ]
        
        print(f"🏗️ Creating project: {self.project_name}")
        print("=" * 50)
        
        # Create main project directory
        self.project_path.mkdir(exist_ok=True)
        self.created_dirs.append(str(self.project_path))
        
        # Create subdirectories
        for directory in directories:
            dir_path = self.project_path / directory
            dir_path.mkdir(exist_ok=True)
            self.created_dirs.append(str(dir_path))
            print(f"✅ Created directory: {directory}/")
    
    def create_empty_file(self, filepath, description=""):
        """Create an empty file with comment header"""
        full_path = self.project_path / filepath
        
        # Ensure parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with appropriate comment based on extension
        file_ext = full_path.suffix
        
        if file_ext == '.py':
            content = f'"""\n{description}\n\nTODO: Add implementation here\n"""\n\n# Add your code here\n'
        elif file_ext == '.yaml' or file_ext == '.yml':
            content = f'# {description}\n# TODO: Add configuration here\n\n'
        elif file_ext == '.md':
            content = f'# {description}\n\nTODO: Add documentation here\n'
        elif file_ext == '.txt':
            content = f'# {description}\n# TODO: Add content here\n\n'
        elif file_ext == '.ipynb':
            content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ''' + description + '''\\n",
    "\\n",
    "TODO: Add notebook content here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
        else:
            content = f'# {description}\n# TODO: Add content here\n'
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.created_files.append(str(full_path))
        print(f"✅ Created file: {filepath} {description}")
    
    def create_all_files(self):
        """Create all project files"""
        print("\n📝 Creating project files...")
        
        # Source files
        self.create_empty_file("src/__init__.py", "(Package initialization)")
        self.create_empty_file("src/model.py", "(CNN architecture)")
        self.create_empty_file("src/dataset.py", "(Dataset handling)")
        self.create_empty_file("src/train.py", "(Training pipeline)")
        self.create_empty_file("src/utils.py", "(Utility functions)")
        self.create_empty_file("src/visualize.py", "(Visualization tools)")
        
        # Main execution file
        self.create_empty_file("main.py", "(Main execution script)")
        
        # Configuration
        self.create_empty_file("configs/config.yaml", "(Configuration file)")
        
        # Dependencies and documentation
        self.create_empty_file("requirements.txt", "(Dependencies)")
        self.create_empty_file("README.md", "(Project documentation)")
        
        # Jupyter notebooks
        self.create_empty_file("notebooks/exploration.ipynb", "Data exploration")
        self.create_empty_file("notebooks/experiments.ipynb", "Model experiments")
        self.create_empty_file("notebooks/analysis.ipynb", "Results analysis")
        
        # Test files
        self.create_empty_file("tests/__init__.py", "(Test package)")
        self.create_empty_file("tests/test_model.py", "(Model tests)")
        self.create_empty_file("tests/test_dataset.py", "(Dataset tests)")
        self.create_empty_file("tests/test_train.py", "(Training tests)")
        
        # Placeholder files for directories (gitkeep style)
        self.create_empty_file("data/.gitkeep", "(Dataset storage directory)")
        self.create_empty_file("checkpoints/.gitkeep", "(Model checkpoints directory)")
        self.create_empty_file("results/.gitkeep", "(Results and plots directory)")
        self.create_empty_file("logs/.gitkeep", "(Training logs directory)")
        
        # Additional useful files
        self.create_empty_file(".gitignore", "(Git ignore file)")
        self.create_empty_file("setup.py", "(Package setup file)")
        self.create_empty_file("LICENSE", "(License file)")
    
    def create_gitignore_content(self):
        """Create .gitignore with ML project specific ignores"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Data
data/cifar-10-batches-py/
data/cifar-10-python.tar.gz
*.pkl
*.pickle

# Model checkpoints
checkpoints/*.pth
checkpoints/*.pt

# Results
results/*.png
results/*.jpg
results/*.pdf

# Logs
logs/*.log
logs/*.txt

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
"""
        
        gitignore_path = self.project_path / ".gitignore"
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        
        print("✅ Created file: .gitignore (Git ignore rules)")
    
    def print_file_map(self):
        """Print a map of what code goes in which file"""
        print("\n" + "="*60)
        print("📋 FILE MAPPING - WHAT CODE GOES WHERE")
        print("="*60)
        
        file_mapping = {
            "src/__init__.py": "Package initialization and imports",
            "src/model.py": "CIFAR10CNN class - CNN architecture implementation", 
            "src/dataset.py": "CIFAR10CustomDataset class and data transforms",
            "src/train.py": "Training functions: train_epoch, validate_epoch, train_model",
            "src/utils.py": "MetricsTracker, ModelCheckpoint, utility functions",
            "src/visualize.py": "Plotting functions: training history, predictions, confusion matrix",
            "main.py": "Main execution script that orchestrates training",
            "configs/config.yaml": "YAML configuration with hyperparameters",
            "requirements.txt": "List of Python dependencies",
            "README.md": "Project documentation and setup instructions"
        }
        
        for filepath, description in file_mapping.items():
            print(f"📄 {filepath:<25} → {description}")
        
        print("\n" + "="*60)
        print("💡 COPY CODE STRATEGY:")
        print("="*60)
        print("1. Copy each code section from the artifact to its corresponding file")
        print("2. Replace the TODO comments with the actual implementation")
        print("3. Start with src/model.py, then src/dataset.py, etc.")
        print("4. Test each module individually before running main.py")
    
    def print_summary(self):
        """Print project creation summary"""
        print("\n" + "="*60)
        print("🎉 PROJECT STRUCTURE CREATED!")
        print("="*60)
        print(f"📁 Project Name: {self.project_name}")
        print(f"📂 Directories: {len(self.created_dirs)}")
        print(f"📄 Files: {len(self.created_files)}")
        
        print("\n🚀 NEXT STEPS:")
        print("-" * 30)
        print(f"1. cd {self.project_name}")
        print("2. Copy code from the provided code sections into corresponding files")
        print("3. pip install -r requirements.txt")
        print("4. python main.py")
        
        print(f"\n📁 PROJECT STRUCTURE CREATED:")
        print("-" * 30)
        print(f"{self.project_name}/")
        print("├── src/")
        print("│   ├── __init__.py")
        print("│   ├── model.py         ← Copy CIFAR10CNN class here")
        print("│   ├── dataset.py       ← Copy dataset and transforms here") 
        print("│   ├── train.py         ← Copy training functions here")
        print("│   ├── utils.py         ← Copy utilities here")
        print("│   └── visualize.py     ← Copy visualization functions here")
        print("├── notebooks/")
        print("│   ├── exploration.ipynb")
        print("│   ├── experiments.ipynb")
        print("│   └── analysis.ipynb")
        print("├── configs/")
        print("│   └── config.yaml      ← Copy YAML configuration here")
        print("├── tests/")
        print("│   ├── __init__.py")
        print("│   ├── test_model.py")
        print("│   ├── test_dataset.py")
        print("│   └── test_train.py")
        print("├── checkpoints/         (empty - for model saves)")
        print("├── results/             (empty - for plots)")
        print("├── logs/                (empty - for training logs)")
        print("├── data/                (empty - for CIFAR-10 dataset)")
        print("├── main.py              ← Copy main execution script here")
        print("├── requirements.txt     ← Copy dependencies list here")
        print("├── README.md            ← Copy documentation here")
        print("├── .gitignore")
        print("├── setup.py")
        print("└── LICENSE")
        
        print("\n✨ All directories and files created successfully!")
        print(f"📍 Location: ./{self.project_name}/")
        print("📝 Ready for you to copy the code into each file!")
    
    def generate_project(self):
        """Generate complete project structure"""
        try:
            self.create_directory_structure()
            self.create_all_files()
            self.create_gitignore_content()
            self.print_file_map()
            self.print_summary()
            return True
        except Exception as e:
            print(f"❌ Error creating project: {e}")
            return False


def get_project_name():
    """Interactive function to get project name from user"""
    print("🤖 CIFAR-10 CNN Project Structure Generator")
    print("📁 Creating directories and empty files only")
    print("=" * 50)
    
    # Ask for project name
    while True:
        project_name = input("📝 Enter project folder name (default: cifar10-cnn-classifier): ").strip()
        
        # Use default if empty
        if not project_name:
            project_name = "cifar10-cnn-classifier"
        
        # Validate project name
        if not project_name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            print("❌ Invalid project name. Use only letters, numbers, hyphens, and underscores.")
            continue
        
        # Check if directory already exists
        if os.path.exists(project_name):
            print(f"⚠️  Directory '{project_name}' already exists!")
            response = input("   Continue anyway? (y/N): ").strip().lower()
            if response == 'y' or response == 'yes':
                break
            else:
                continue
        else:
            break
    
    return project_name


def main():
    """Main function to run the project generator"""
    # Interactive mode - ask for project name
    if len(sys.argv) == 1:
        project_name = get_project_name()
    else:
        # Command line argument mode
        parser = argparse.ArgumentParser(description='Create CIFAR-10 CNN project structure with empty files')
        parser.add_argument('project_name', nargs='?', default='cifar10-cnn-classifier',
                           help='Name of the project directory (default: cifar10-cnn-classifier)')
        
        args = parser.parse_args()
        project_name = args.project_name
        
        print("🤖 CIFAR-10 CNN Project Structure Generator")
        print("📁 Creating directories and empty files only")
        print("=" * 50)
        print(f"📁 Using project name: {project_name}")
        
        # Check if project directory already exists
        if os.path.exists(project_name):
            response = input(f"⚠️  Directory '{project_name}' already exists. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Project generation cancelled.")
                return
    
    # Generate project
    print(f"\n🏗️ Creating project structure for: {project_name}")
    generator = ProjectStructureGenerator(project_name)
    success = generator.generate_project()
    
    if success:
        print(f"\n🎯 SUCCESS! Empty project structure created in '{project_name}/'")
        print("📝 Now you can manually copy the code into each file!")
        print(f"💡 Next: cd {project_name}")
    else:
        print(f"\n💥 FAILED! There was an error creating the project structure.")


if __name__ == "__main__":
    main()