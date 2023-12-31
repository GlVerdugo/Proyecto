# Virtual Environments
Alumno: Guadalupe López Verdugo
Matricula:A01688491

Follow the instructions below to do the activity.
### Run the existing notebook
1. Create a virtual environment with Python 3.10.9 (windows)
    * Create venv
        ```
        py3.10 -m venv venv310
        ```

    * Activate the virtual environment
    * Go to the SCRIPTS folder and run the file activate

        ```
        1. cd venv310
        2. cd Scripts
        3. activate
        ```

2. Install libraries
    Run the following command to install the other libraries.

    ```bash
    pip install -r ACTIVIDAD SESION3\requirements.txt
    ```
    Verify the installation with this command:
    ```bash
    pip freeze
    ```
    Output:
    <details open>
    <summary>List of packages, click to collapse</summary>
  
        arrow==1.2.3
        asttokens==2.2.1
        backcall==0.2.0 
        binaryornot==0.4.4
        certifi==2023.7.22
        chardet==5.2.0
        charset-normalizer==3.2.0
        click==8.1.6
        colorama==0.4.6
        comm==0.1.4
        contourpy==1.1.0
        cookiecutter==2.3.0
        cycler==0.11.0
        debugpy==1.6.7
        decorator==5.1.1
        exceptiongroup==1.1.2
        executing==1.2.0
        fonttools==4.42.0
        idna==3.4
        iniconfig==2.0.0
        ipykernel==6.25.1
        ipython==8.14.0
        jedi==0.19.0
        Jinja2==3.1.2
        joblib==1.3.1
        jupyter_client==8.3.0
        jupyter_core==5.3.1
        kiwisolver==1.4.4
        markdown-it-py==3.0.0
        MarkupSafe==2.1.3
        matplotlib==3.7.2
        matplotlib-inline==0.1.6
        mdurl==0.1.2
        nest-asyncio==1.5.7
        numpy==1.25.2
        packaging==23.1
        pandas==2.0.3
        parso==0.8.3
        pickleshare==0.7.5
        Pillow==10.0.0
        platformdirs==3.10.0
        pluggy==1.2.0
        prompt-toolkit==3.0.39
        psutil==5.9.5
        pure-eval==0.2.2
        Pygments==2.16.1
        pyparsing==3.0.9
        pytest==7.4.0
        python-dateutil==2.8.2
        python-slugify==8.0.1
        pytz==2023.3
        pywin32==306
        PyYAML==6.0.1
        pyzmq==25.1.0
        requests==2.31.0
        rich==13.5.2
        scikit-learn==1.1.1
        scipy==1.11.1
        seaborn==0.12.2
        six==1.16.0
        stack-data==0.6.2
        text-unidecode==1.3
        threadpoolctl==3.2.0
        tomli==2.0.1
        tornado==6.3.2
        traitlets==5.9.0
        tzdata==2023.3
        urllib3==2.0.4
        wcwidth==0.2.6
        
    </details>
    

4. Open the `heart-disease-prediction.ipynb` notebook and click on `Run All`. 
    > **IMPORTANT!**  
    Do not forget to select the Python 3.9.10 kernel you have already created.
    ```
**Congrats, the notebook is running in a virtual environment with Python 3.10.9!**
