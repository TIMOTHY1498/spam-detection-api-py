# spam-detection-api-py

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)

## Installation

First install all the required package via pip

- pandas
- sklearn
- nltk
- uvicorn
- pydantic
- fastapi

Second, download the source code via this github page

Lastly, run the file `main.py` to start hosting the API by this command
```shell
uvicorn main:app --reload
```

## Usage
⚠️ **Disclaimer**: 
You need to change the dataset for training. You can change the hosted training data link via:
```py
url = '' # List message that used for the AI Training, please use your own!
```

By the way, the hosted API link is ```23.88.73.88:35922 (url having issue)```  

To use the API, Create a post request as same as this example: 

```json
{
    "message" : "Message to check Here!"
}
```

And the API will respond:

```json
{
    "isSpam" : true // or false
}
```

## Credits

Credits to:

- Microsoft Copilot
- Me (TIMOTHY1498)
