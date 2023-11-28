# OpenAI Q&A Pair Generator

Deploy this function on your server, and you will get a service that uses OpenAI's GPT models to generate question and answer pairs from text provided via HTTP POST requests.

## Prerequisites

* You will need to bring your own [OpenAI API key](https://openai.com/blog/openai-api). If you do not already have one, [sign up here](https://platform.openai.com/signup).

* Set the `OPENAI_API_KEY` environment variable to your API key value.
* Optional: set the `SYS_PROMPT` environment variable to the system prompt for QA generation.

You'll get a unique webhook URL after your flow function has been successfully deployed.

## Give it a try

Send a POST request with the input text to your webhook URL:

```bash
curl -X POST https://code.flows.network/webhook/htObCFjbGAI4kolgmRRk -H "Content-Type: text/plain" --data-binary "@test.txt"
```

You'll receive a CSV response with Q&A pairs derived from the text you submitted. The `test.txt` file has 4 sections of text separated by blank lines. The flow function should return about 15 QA pairs for each section of text.

The flow function would time out after about 20 minutes. That translates to about 40 sections of input text. If you have more text, you can segment the input into multiple files and run this flow function repeatedly, and then join the result CSV data together.
