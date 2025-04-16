Document OCR for Typed Documents(Not for Handwritten Docs)

- Create Virtual env using conda
  Make sure you have selected the conda python interpretor first(ctrl+shift+p - select python interpretor)

  # for creating env

          - conda create --prefix ./envs                           # this creates env in the project folder
          # conda env create --prefix ./envs -f requirements.yml   # if you want requirements

  # for activating env

          - conda config --set env_prompt '({name})'                # helps to remove the long filepath prefix
          - conda activate ./envs

  # Common commands

          - conda info --envs
          - conda list --explicit                                    # lists the installed packages

  # for deactivating env

          - conda config --set env_prompt '({default_env})'
          - conda deactivate
          - conda activate base

To Launch Jupyter notebook, we can use ctrl+shft+p and click "Create: New upyter Notebook".
Before that make sure you have ipykernel installed in the environment you are working with
conda install -p e:\Learning\ML\Learning Practice\Code\Computer Vision\OCR\envs ipykernel --update-deps --force-reinstall

- Required libraries

- Pillow
- OpenCV
- Tesseract app
- PyTesseract
