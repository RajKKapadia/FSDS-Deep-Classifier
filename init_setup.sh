echo [$(date)]: "START" 
echo [$(date)]: "creating env with python 3.8 version" 
python -m venv venv
echo [$(date)]: "activating the environment" 
source venv/bin/activate
echo [$(date)]: "updating pip version"
python -m pip install --upgrade pip
echo [$(date)]: "installing the dev requirements" 
pip install -r requirements_dev.txt
echo [$(date)]: "END"