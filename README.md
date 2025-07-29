# ML Project Structure Generator

A Python script that automatically creates professional machine learning project structures with proper directory organization and empty template files. Perfect for starting new ML projects with industry-standard layouts.

## 🎯 Features

- **Interactive Project Naming** - Prompts for custom project names
- **Professional Structure** - Industry-standard ML project layout
- **Empty Template Files** - Pre-created files with TODO comments
- **Multiple Usage Modes** - Interactive or command-line execution
- **Input Validation** - Ensures valid project names and handles conflicts
- **Comprehensive Setup** - Includes configs, tests, notebooks, and documentation

## 📁 Generated Project Structure

```
your-project-name/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── model.py           # Model architecture
│   ├── dataset.py         # Dataset handling
│   ├── train.py          # Training pipeline
│   ├── utils.py          # Utility functions
│   └── visualize.py      # Visualization tools
├── notebooks/             # Jupyter notebooks
│   ├── exploration.ipynb # Data exploration
│   ├── experiments.ipynb # Model experiments
│   └── analysis.ipynb    # Results analysis
├── configs/               # Configuration files
│   └── config.yaml       # Main configuration
├── tests/                 # Unit tests
│   ├── __init__.py       # Test package
│   ├── test_model.py     # Model tests
│   ├── test_dataset.py   # Dataset tests
│   └── test_train.py     # Training tests
├── checkpoints/          # Model checkpoints (empty)
├── results/             # Training results and plots (empty)
├── logs/                # Training logs (empty)
├── data/                # Dataset storage (empty)
├── main.py             # Main execution script
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
├── .gitignore         # Git ignore rules
├── setup.py           # Package setup
└── LICENSE            # License file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.6+
- No additional dependencies required

### Usage

#### Interactive Mode (Recommended)
```bash
python Create_ML_Project_Structure.py
```
The script will prompt you for a project name:
```
📝 Enter project folder name (default: cifar10-cnn-classifier): my-awesome-project
```

#### Command Line Mode
```bash
python Create_ML_Project_Structure.py my-project-name
```

#### Using Default Name
```bash
python Create_ML_Project_Structure.py
# Press Enter to use default: cifar10-cnn-classifier
```

## 📋 What Gets Created

### Source Code Structure
- **`src/model.py`** - Model architecture implementation
- **`src/dataset.py`** - Data loading and preprocessing
- **`src/train.py`** - Training and validation logic
- **`src/utils.py`** - Helper functions and utilities
- **`src/visualize.py`** - Plotting and visualization functions

### Configuration & Setup
- **`configs/config.yaml`** - Centralized configuration management
- **`requirements.txt`** - Python dependencies
- **`.gitignore`** - ML-specific ignore patterns
- **`setup.py`** - Package installation setup

### Development Tools
- **`tests/`** - Unit test templates
- **`notebooks/`** - Jupyter notebook templates
- **`README.md`** - Project documentation template

### Data & Results
- **`data/`** - Dataset storage directory
- **`checkpoints/`** - Model checkpoint storage
- **`results/`** - Plots and analysis results
- **`logs/`** - Training logs and metrics

## 🛠️ Customization

### Adapting for Different ML Projects

This structure works for various ML projects:

#### Computer Vision Projects
- Use `src/model.py` for CNN/Vision Transformer architectures
- Use `src/dataset.py` for image preprocessing and augmentation
- Store datasets in `data/` directory

#### NLP Projects
- Implement transformer models in `src/model.py`
- Handle text preprocessing in `src/dataset.py`
- Store tokenizers and embeddings in `data/`

#### Time Series Projects
- Create LSTM/GRU models in `src/model.py`
- Implement sequence preprocessing in `src/dataset.py`
- Store time series data in `data/`

### Modifying the Generator

To customize the structure for your needs:

1. **Add New Directories:**
```python
directories = [
    "src", "notebooks", "configs", "tests",
    "checkpoints", "results", "logs", "data",
    "your_custom_dir"  # Add your directory here
]
```

2. **Add New Files:**
```python
self.create_empty_file("your_new_file.py", "(Your file description)")
```

3. **Customize File Templates:**
Modify the `create_empty_file()` method to add custom templates.

## 📚 Usage Examples

### Example 1: Computer Vision Project
```bash
python Create_ML_Project_Structure.py image-classifier
cd image-classifier
# Copy your CNN implementation to src/model.py
# Copy dataset handling to src/dataset.py
# Copy training loop to src/train.py
```

### Example 2: NLP Project
```bash
python Create_ML_Project_Structure.py sentiment-analysis
cd sentiment-analysis
# Implement transformer model in src/model.py
# Add text preprocessing to src/dataset.py
# Configure hyperparameters in configs/config.yaml
```

### Example 3: Research Project
```bash
python Create_ML_Project_Structure.py research-experiment
cd research-experiment
# Use notebooks/ for exploration
# Implement models in src/
# Track experiments in logs/
```

## 🔧 Advanced Features

### Git Integration
The generated `.gitignore` includes ML-specific patterns:
- Data files and datasets
- Model checkpoints
- Results and plots
- Jupyter notebook checkpoints
- Python cache files

### Testing Framework
Pre-created test files for:
- Model architecture validation
- Dataset loading verification
- Training pipeline testing

### Configuration Management
YAML-based configuration for:
- Model hyperparameters
- Training settings
- Data paths
- Experiment tracking

## 📖 Best Practices

### Project Organization
1. **Keep `src/` clean** - Only source code, no data or results
2. **Use `configs/`** - Centralize all hyperparameters
3. **Document in `notebooks/`** - Use for exploration and analysis
4. **Test everything** - Write tests in `tests/` directory

### Development Workflow
1. **Start with exploration** - Use `notebooks/exploration.ipynb`
2. **Implement incrementally** - Build `src/` modules step by step
3. **Test frequently** - Run tests after each major change
4. **Track experiments** - Use `logs/` and `results/`

### Version Control
```bash
cd your-project-name
git init
git add .
git commit -m "Initial project structure"
git remote add origin your-repo-url
git push -u origin main
```

## 🤝 Contributing

### Adding New Templates
1. Fork the repository
2. Add new file templates in `create_empty_file()` method
3. Update the directory structure if needed
4. Test with different project types
5. Submit a pull request

### Reporting Issues
- Use GitHub issues for bug reports
- Include the generated structure output
- Specify your Python version and OS

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by industry-standard ML project structures
- Designed for PyTorch, TensorFlow, and other ML frameworks
- Suitable for research, development, and production projects

## 🔗 Related Resources

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [MLOps Best Practices](https://ml-ops.org/)
- [PyTorch Project Template](https://github.com/pytorch/pytorch-template)

---

**Happy Machine Learning! 🚀**

Made with ❤️ for the ML community. Star ⭐ this repo if it helps your projects!