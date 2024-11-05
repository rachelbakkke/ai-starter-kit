from typing import List, Tuple

import streamlit

from financial_assistant.constants import *
from financial_assistant.streamlit.llm_model import sambanova_llm
from financial_assistant.src.utilities import get_logger
from financial_assistant.streamlit.utilities_app import save_output_callback
from financial_assistant.streamlit.utilities_methods import TOOLS, handle_userinput

logger = get_logger()


def get_yfinance_news() -> None:
    streamlit.markdown('<h2> Financial News Scraping </h2>', unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://uk.finance.yahoo.com/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance News</h3></a>',
        unsafe_allow_html=True,
    )

    user_request = streamlit.text_input(
        label='Enter the yfinance news that you want to retrieve for given companies. '
        f':sparkles: :violet[{DEFAULT_RAG_QUERY}]',
        key='yahoo_news',
        placeholder='E.g. ' + DEFAULT_RAG_QUERY,
    )

    # Retrieve news
    if streamlit.button('Retrieve News'):
        if len(user_request) == 0:
            logger.error('No query entered.')
            streamlit.error('No query entered.')
        else:
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                if user_request is not None:
                    answer, url_list = handle_yfinance_news(user_request)
                else:
                    raise ValueError('No input provided')

            if answer is not None:
                content = answer + '\n\n'.join(url_list)

                # Save the query and answer to the history text file
                save_output_callback(content, HISTORY_PATH, user_request)

                # Save the query and answer to the Yahoo Finance News text file
                if streamlit.button(
                    'Save Answer',
                    on_click=save_output_callback,
                    args=(content, YFINANCE_NEWS_PATH, user_request),
                ):
                    pass


def handle_yfinance_news(user_question: str) -> Tuple[str, List[str]]:
    """
    Handle the user request for the Yahoo News data.

    Args:
        user_question: The user input question that is used to retrieve the Yahoo Finance News data.

    Returns:
        A tuple containing the following pair:
            1. The answer to the user query.
            2. A list of links to articles that have been used for retrieval to answer the user query.

    Raises:
        Exception: If the LLM response does not conform to the expected return type.
    """
    # Declare the permitted tools for function calling
    tools = [
        'scrape_yahoo_finance_news',
    ]

    # Set the tools for the LLM to use
    sambanova_llm.tools = [TOOLS[name] for name in tools]

    user_request = f"""
        You are an expert in the stock market.
        Please answer the following question, which may be general or related to a specific list of companies:
        {user_question}

        First, extract the company (or companies) from the user query, if applicable.
        Then, retrieve the relevant news articles by web scraping Yahoo Finance,
        and then provide the answer to the user.
    """

    # Call the LLM on the user request with the attached tools
    response = handle_userinput(user_question, user_request)

    # Reset tools
    sambanova_llm.tools = None

    # Check the final answer of the LLM
    if (
        not isinstance(response, tuple)
        or len(response) != 2
        or not isinstance(response[0], str)
        or not isinstance(response[1], list)
        or not all(isinstance(item, str) for item in response[1])
    ):
        raise Exception(f'Invalid response: {response}.')

    return response
