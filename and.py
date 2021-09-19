

from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")




def main(data, modelName, plotName, eta, epochs):
    """
    Main code 
    
    """
    df = pd.DataFrame(data)
    #print(df)
    logging.info(f"This is actual dataframe{df}")
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename=modelName)
    save_plot(df, plotName, model)

if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    try:
        logging.info(">>>>>>>>>>>>STARTING TRAINING>>>>>>>>>")
        main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)
        logging.info("<<<<<<<TRAINING DONE SUCCESSFULLY<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e