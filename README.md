## running the app

It’s a Python Streamlit application. Here’s a brief summary of what it does:
The application allows a user to upload a PDF file.
The text is extracted from the PDF file and the number of tokens in the text is calculated.
A question related to the PDF content can be asked by the user.
The application uses OpenAI’s GPT-3 model to find the most relevant part of the PDF to the question and then answers the question based on that part.
It calculates the cost based on the number of tokens processed by the model and displays this to the user.


locally
```
$ poetry run streamlit run app.py  
```


Docker

```
$ docker-compose up -d
```