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
- fastapi
- pydantic

Second, download the source code via this github page

Lastly, run the file `main.py` to start hosting the API by this command
```shell
uvicorn main:app --reload
```

## Usage

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

- Microsofft Copilot
- Me (TIMOTHY1498)