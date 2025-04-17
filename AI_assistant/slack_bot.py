import logging
import os

from dotenv import find_dotenv, load_dotenv
from flask import Flask, jsonify, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_functions import draft_email
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID")
print(SLACK_BOT_TOKEN)
print(SLACK_SIGNING_SECRET)
print(SLACK_BOT_USER_ID)
# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

signature_verifier = SignatureVerifier(SLACK_SIGNING_SECRET)
# Initialize the Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=SLACK_BOT_TOKEN)
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")


def my_function(text):
    """
    Custom function to process the text and return a response.
    In this example, the function converts the input text to uppercase.

    Args:
        text (str): The input text to process.

    Returns:
        str: The processed text.
    """
    response = text.upper()
    return response


@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    logger.debug("MENTION DETECTED")
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    response = draft_email(text)
    say(response)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    logger.debug("Received request to /slack/events")
    logger.debug(f"Headers: {dict(request.headers)}")

    request_body = request.get_data().decode("utf-8")
    logger.debug(f"Request body: {request_body}")

    # Log environment variables (be careful not to log actual secrets in production)

    # Verify the request is coming from Slack
    try:
        is_valid = signature_verifier.is_valid_request(
            request.get_data(), request.headers
        )
        logger.debug(f"Request validation result: {is_valid}")

        if not is_valid:
            logger.warning("Invalid request signature")
            return "Invalid request", 403
    except Exception as e:
        logger.error(f"Error validating request: {str(e)}")
        return "Verification error", 403

    # Handle URL verification challenge
    if request.json and "challenge" in request.json:
        logger.info("Handling URL verification challenge")
        return jsonify({"challenge": request.json["challenge"]})

    # Process the event
    try:
        event_data = request.json
        logger.info(f"Processing event: {event_data}")
        # Handle your event logic here

        return handler.handle(request)
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        return "Error processing event", 500


@flask_app.route("/", methods=["GET"])
def health_check():
    return "Bot server is running!", 200


if __name__ == "__main__":
    logger.info("Starting Slack bot server on port 3345")
    flask_app.run(debug=True, port=3345)
