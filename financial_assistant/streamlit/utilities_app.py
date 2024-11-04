import datetime
import json
import os
import pathlib
import re
import shutil
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import pandas
import pandasai
import schedule
import streamlit
from matplotlib.figure import Figure
from streamlit.elements.widgets.time_widgets import DateWidgetReturn

from financial_assistant.constants import *
from financial_assistant.src.utilities import get_logger
from utils.visual.env_utils import initialize_env_variables

logger = get_logger()


def initialize_session(
    session_state: streamlit.runtime.state.session_state_proxy.SessionStateProxy,
    prod_mode: bool = False,
    cache_dir: str = CACHE_DIR,
) -> None:
    """Initialize the Streamlit `session_state`."""

    # Initialize the production mode
    if 'prod_mode' not in session_state:
        session_state.prod_mode = prod_mode

    # Initialize credentials
    initialize_env_variables(prod_mode)

    # Initialize the chat history
    if 'chat_history' not in session_state:
        session_state.chat_history = list()

    # Launch time
    if 'launch_time' not in session_state:
        session_state.launch_time = datetime.datetime.now()

    # Delete pandasai cache
    try:
        pandasai.clear_cache()
    except:
        pass
    delete_temp_dir(temp_dir=PANDASAI_CACHE, verbose=False)


def submit_sec_edgar_details() -> None:
    """Add the SEC-EDGAR details to the session state."""

    key = 'sidebar-sec-edgar'
    sec_edgar_help = """Must provide organization and email address
        to comply with the SEC Edgar's downloading fair access
        <a href="https://www.sec.gov/os/webmaster-faq#code-support" target="_blank">policy</a>.
    """
    if os.getenv('SEC_API_ORGANIZATION') is None or os.getenv('SEC_API_EMAIL') is None:
        streamlit.markdown(sec_edgar_help, unsafe_allow_html=True)

    # Populate SEC-EDGAR credentials
    if os.getenv('SEC_API_ORGANIZATION') is None:
        os.environ['SEC_API_ORGANIZATION'] = streamlit.text_input(
            'For SEC-EDGAR: <your organization>', None, key=key + '-organization'
        )  # type: ignore

    if os.environ['SEC_API_EMAIL'] is None:
        os.environ['SEC_API_EMAIL'] = streamlit.text_input(
            'For SEC-EDGAR: <user@email_provider.com>', None, key=key + '-email'
        )
    # Save button
    if os.getenv('SEC_API_ORGANIZATION') is None or os.environ['SEC_API_EMAIL'] is None:
        if streamlit.button('Save SEC EDGAR details', key=key + '-button'):
            if os.getenv('SEC_API_ORGANIZATION') is not None and os.environ['SEC_API_EMAIL'] is not None:
                streamlit.success('SEC EDGAR details saved successfully!')
        else:
            streamlit.warning('Please enter organization and email.')


def save_historical_price_callback(
    user_query: str,
    symbol_list: List[str],
    data: pandas.DataFrame,
    fig: Figure,
    start_date: DateWidgetReturn,
    end_date: DateWidgetReturn,
    save_path: Optional[str] = None,
) -> None:
    """Save dataframe and figure callback for streamlit button."""

    # Derive the directory name
    dir_name = HISTORY_FIGURES_DIR

    # Create the directory for storing historical price data
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Derive the filename
    suffix = f'stock_data_{"_".join(symbol_list)}_{start_date}_{end_date}'
    path_csv = os.path.join(dir_name, f'{suffix}.csv')
    path_png = os.path.join(dir_name, f'{suffix}.png')

    # Write the dataframe to a csv file
    data.to_csv(path_csv, index=True)

    # Save the plots as png images
    fig.savefig(path_png, bbox_inches='tight')

    # Compose the content including the user query and the filename
    content = '\n\n' + user_query + '\n\n' + f'{path_png}' + '\n\n'

    # Save the content path to a file
    if save_path is not None:
        save_output_callback(content, save_path)


def save_output_callback(
    response: str | List[str] | Dict[str, str],
    save_path: str,
    user_request: Optional[str] = None,
) -> None:
    """Save the output callback for streamlit button."""

    # Check the inputs
    if not isinstance(response, (str, list, dict, tuple, pandas.Series, pandas.DataFrame)):
        raise TypeError(
            'Response must be a string, a list, a dictionary, a tuple. a series, or a dataframe. '
            f'Got type {type(response)}'
        )
    if not isinstance(save_path, str):
        raise TypeError('Save path must be a string.')
    if not isinstance(user_request, (str, type(None))):
        raise TypeError('User request must be a string.')

    # Specify the filename
    filename = save_path

    # Opening space
    with open(filename, 'a') as text_file:
        text_file.write('\n\n')

    # Add the user query
    if user_request is not None:
        with open(filename, 'a') as text_file:
            text_file.write(user_request)
            text_file.write('\n\n')

    # If the response is a set, convert it to a list
    if isinstance(response, set):
        response = list(response)

    # If the reponse is a simple element, save it as it is
    if isinstance(response, (str, float, int, pandas.Series, pandas.DataFrame)):
        # Write the string to a txt file
        save_simple_output(response, filename)

    # If the response is a list, split it up and write each element individually
    elif isinstance(response, list):
        for elem in response:
            save_simple_output(elem + '\n', filename)

    # If the response is a dict, split it up and write pair of key and value individually
    elif isinstance(response, dict):
        for key, value in response.items():
            with open(filename, 'a') as text_file:
                if isinstance(value, (str, float, int)):
                    save_simple_output(value + '\n', filename)
                elif isinstance(value, list):
                    save_simple_output(', '.join([str(item) for item in value]) + '.' + '\n', filename)
                elif isinstance(value, (pandas.Series, pandas.DataFrame)):
                    save_simple_output(value, filename)
                else:
                    save_simple_output(value, filename)

    # If the response is a tuple, split it up and write each element individually
    elif isinstance(response, tuple):
        for elem in response:
            save_simple_output(response + '\n', filename)

    elif isinstance(response, (pandas.Series, pandas.DataFrame0)):
        save_simple_output(response, filename)

    else:
        raise ValueError('Invalid response type')

    # Closing space
    with open(filename, 'a') as text_file:
        text_file.write('\n\n')


def save_simple_output(
    response: Any,
    filename: str,
) -> None:
    """
    Saves the response to a file in the specified path.

    Args:
        response: The response to be saved.
        filename: The path to save the response in.
    """

    # If the response is a string or number, write it directly into the text file
    if isinstance(response, (str, float, int)):
        if isinstance(response, float):
            response = round(response, 2)
        with open(filename, 'a') as text_file:
            text_file.write(str(response))

    # If the response is a Series or DataFrame, convert it to a dictionary and then dump it to a JSON file
    elif isinstance(response, (pandas.Series, pandas.DataFrame)):
        # Convert the response to json
        json_string = response.to_json(orient='records')

        # Write the json string to a txt file
        with open(filename, 'a') as text_file:
            text_file.write(json_string)
    else:
        try:
            # Write the dataframe to a CSV file
            response_dict = response.to_dict()

            # Convert the dictionary into a string
            stripped_string = dump_stripped_json(response_dict)

            # Write the stripped string to a txt file
            with open(filename, 'a') as text_file:
                text_file.write(stripped_string)
        except:
            streamlit.warning('Could not save the response.')

    # Spaces between elements or closing space
    with open(filename, 'a') as text_file:
        text_file.write('\n\n')


def dump_stripped_json(data: Any, indent: int = 2) -> str:
    """
    Dump a JSON-serializable object to a string, stripping parentheses and apostrophes.

    Args:
        data: The JSON-serializable object (dict or list) to dump.
        indent: The indentation level for pretty-printing (default is 2).

    Returns:
        A string representation of the JSON data without parentheses and apostrophes.
    """
    # Dump the data to a JSON string
    json_string = json.dumps(data, indent=indent)

    # Remove parentheses and apostrophes
    stripped_string = re.sub(r'["\'{}]', '', json_string)

    # Remove any empty lines that might have been created by stripping
    stripped_string = '\n'.join(line for line in stripped_string.split('\n') if line.strip())

    return stripped_string


def list_files_in_directory(directory: str) -> List[str]:
    """List all files in the given directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def list_directory(directory: str) -> Tuple[List[str], List[str]]:
    """List subdirectories and files in the given directory."""
    subdirectories = []
    files = []

    if os.getcwd() == repo_dir:
        pass
    elif os.getcwd() == kit_dir:
        os.chdir(os.path.realpath(os.path.dirname(os.getcwd())))

    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            subdirectories.append(name)
        else:
            files.append(name)
    return subdirectories, files


def display_directory_contents(path: str, default_path: str) -> None:
    """Display subdirectories and files in the current path, up to the default path."""

    # Check that `path` is contained relative to `default_path`
    if not pathlib.Path(path).is_relative_to(pathlib.Path(default_path)):
        path = default_path
        streamlit.session_state.current_path = default_path

    subdirectories, files = list_directory(path)

    dir_name = Path(path).name
    if dir_name.startswith('cache'):
        dir_name = 'cache'

    if subdirectories:
        for idx, subdir in enumerate(subdirectories):
            if streamlit.sidebar.button(f'📁 {subdir}', key=f'{subdir}_{idx}'):
                files_subdir = list_files_in_directory(os.path.join(path, subdir))
                for file in files_subdir:
                    download_file(
                        os.path.join(streamlit.session_state.current_path, subdir, file), key=file + '_recursion'
                    )

                # Recursion
                display_directory_contents(os.path.join(streamlit.session_state.current_path, subdir), default_path)

    if files and dir_name.startswith('cache'):
        for file in files:
            download_file(os.path.join(path, file), key=file)

    if len(subdirectories + files) == 0:
        streamlit.write('No files found')

    return


def clear_directory(directory: str, delete_subdirectories: bool = False) -> None:
    """Delete all files and optionally subdirectories in the given directory."""

    try:
        if not os.path.exists(directory):
            logger.warning(f'Directory does not exist: {directory}')
            return

        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    # Recurse into subdirectory
                    clear_directory(item_path, delete_subdirectories)
                    if delete_subdirectories:
                        shutil.rmtree(item_path)
            except Exception as e:
                logger.warning(f'Error deleting {item_path}: {e}')
    except Exception as e:
        logger.warning(f'Error processing directory {directory}: {e}')


def clear_cache(delete: bool = False, verbose: bool = False) -> None:
    """Clear and/or delete the cache."""

    try:
        cache_dir = CACHE_DIR

        if not os.path.exists(cache_dir):
            if verbose:
                logger.warning(f'Cache directory does not exist: {Path(cache_dir).name}')
            return

        # Clear the cache directory recursively
        clear_directory(cache_dir, delete)

    except Exception as e:
        logger.warning(f'Error clearing cache directory {Path(cache_dir).name}: {e}')

    if delete:
        try:
            shutil.rmtree(cache_dir)
            if verbose:
                logger.info(f'Successfully deleted cache directory: {Path(cache_dir).name}')
        except Exception as e:
            logger.warning(f'Error deleting cache directory {Path(cache_dir).name}: {e}')


def download_file(filename: str, key: Optional[str] = None) -> None:
    """Add a button to download the file."""

    # Extract the format from the filename
    format = Path(filename).suffix[1:]

    # Extract the correct mime type from the format
    if format == 'txt':
        file_mime = 'text/plain'
    elif format == 'csv':
        file_mime = 'text/csv'
    elif format == 'png':
        file_mime = 'image/png'
    elif format == 'pdf':
        file_mime = 'application/pdf'
    else:
        return
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        streamlit.sidebar.download_button(
            label=Path(filename).name,
            data=data,
            file_name=Path(filename).name,
            mime=file_mime,
            key=key if key is not None else Path(filename).name,
        )
    except Exception as e:
        logger.warning('Error reading file', str(e))
    except FileNotFoundError as e:
        logger.warning('File not found', str(e))


def create_temp_dir_with_subdirs(dir: str, subdirs: List[str] = []) -> None:
    """Create a temporary directory with specified subdirectories."""

    os.makedirs(dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(subdir, exist_ok=True)


def delete_temp_dir(temp_dir: str, verbose: bool = False) -> None:
    """Delete the temporary directory and its contents."""

    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            if verbose:
                logger.info(f'Temporary directory {temp_dir} deleted.')
        except:
            if verbose:
                logger.warning(f'Could not delete temporary directory {temp_dir}.')


def schedule_temp_dir_deletion(temp_dir: str, delay_minutes: int) -> None:
    """Schedule the deletion of the temporary directory after a delay."""

    schedule.every(delay_minutes).minutes.do(delete_temp_dir, temp_dir=temp_dir, verbose=False).tag(temp_dir)

    def run_scheduler() -> None:
        while schedule.get_jobs(temp_dir):
            schedule.run_pending()
            time.sleep(1)

    # Run scheduler in a separate thread to be non-blocking
    Thread(target=run_scheduler, daemon=True).start()


def delete_all_subdirectories(directory: str, exclude: List[str], verbose: bool = False) -> None:
    """
    Delete all subdirectories in the given directory, excluding specified directories.

    Args:
        directory: The parent directory whose subdirectories need to be deleted.
        exclude: A list of subdirectory names to exclude from deletion.
        verbose: Whether to print log messages.
    """
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_path not in exclude and os.path.isdir(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    if verbose:
                        logger.info(f'Successfully deleted directory: {Path(dir_path).name}.')
                except:
                    if verbose:
                        logger.warning(f'Could not delete directory {Path(dir_path).name}.')


def set_css_styles() -> None:
    """Set the CSS style for the streamlit app."""

    streamlit.markdown(
        """
        <style>
        /* General body styling */

        html, body {
            font-size: 16px,
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #e0e0e0;
            background-color: #1e1e1e;
        }

        /* Header styling */
        h1, h2, h3, h4, h5, h6 {
            color: #EE7624;
            margin-bottom: 1em;
        }

        /* Paragraph and text styling */
        p, label {
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 0.5em;
            color: #e0e0e0;
        }

        /* Button styling */
        .stButton > button {
            background-color: #3A8EBA;
            color: white;
            padding: 0.75em 1.5em;
            font-size: 1;
            border: none;
            border-radius: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }

        /* Radio button styling */
        .stRadio > label {
            font-size: 1;
        }
        .stRadio > div > div > label {
            font-size: 1;
            padding: 0.25em 0.75em;
            cursor: pointer;
            color: #e0e0e0;
        }
        .stRadio > div > div {
            margin-bottom: 0.5em;
        }

        /* Input field styling */
        input[type="text"], input[type="date"] select {
            width: 100%;
            padding: 0.75em;
            margin: 0.5em 0 1em 0;
            display: inline-block;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 16px;
            background-color: #2c2c2c;
            color: #e0e0e0;
        }

        /* Checkbox styling */
        .stCheckbox > label {
            font-size: 16px;
        }

        /* Container styling */
        .main {
            font-size: 16px;
            padding: 2em;
            background: #2c2c2c;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 2em;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1e1e1e;
        }
        .css-1d391kg .css-1v3fvcr, .css-1d391kg .css-1l5dyp6 {
            color: #e0e0e0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def get_blue_button_style() -> str:
    """Get the CSS style for a blue button."""
    return """
        button {
            background-color: #2C3E50;
            color: white;
            padding: 0.75em 1.5em;
            font-size: 1;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }"""
