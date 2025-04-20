import logging
import os

from dotenv import find_dotenv, load_dotenv
from flask import Flask, jsonify, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_functions import DEMOGRAPHIC_SCHEMA, RAG_response
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID")
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

# Out of VS code
# create slide deck showing structure < - emphasize scalability and modularity
# create slides for vision of position, working in tandem with bot to identify trends and Business questions automatically
# 3 pillars - universal data access, data cleanliness uniformity, detailed bespoke models
# aggregate queries automatically, build meta assistant that can parse through queries to find trends
# identify new and desired datastreams (from qualtrics etc)
# Modify and transform data to be usable by assistant


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

    # Do some basic keyword detection
    if "intro" in text.lower() or "hello" in text.lower():
        say(
            """
            Hey there! 👋 I’m Scout — your trusty data guide. I'm here to quickly find answers for you from our database.
        Just @Scout with your question to get started!
        Example: "@Scout Does studying lead to better grades?"

        Get more details on the available data by asking me what datasets I have access to.

        If you want to see how I think, start your question with debug!
        Example: "DEBUG Does weekday alchohol usage correlate with absences? @Scout"
            """
        )
    elif "dataset" in text.lower() or "access" in text.lower():
        say(
            f"""
            Here is more information about the dataset I have access to:
            ```
            {DEMOGRAPHIC_SCHEMA}
            ```
        """
        )
    else:

        say("Let me think about that...")
        response = RAG_response(text, say)
        say(response)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():

    # request_body = request.get_data().decode("utf-8")

    # Log environment variables (be careful not to log actual secrets in production)

    # Verify the request is coming from Slack
    try:
        is_valid = signature_verifier.is_valid_request(
            request.get_data(), request.headers
        )

        if not is_valid:
            logger.warning("Invalid request signature")
            return "Invalid request", 403
    except Exception as e:
        logger.error(f"Error validating request: {str(e)}")
        return "Verification error", 403

    # Handle URL verification challenge
    if request.json and "challenge" in request.json:
        return jsonify({"challenge": request.json["challenge"]})

    # Process the event
    try:

        return handler.handle(request)
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        return "Error processing event", 500


if __name__ == "__main__":
    flask_app.run(debug=True, port=3345)
