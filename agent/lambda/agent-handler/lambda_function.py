import json
import datetime
import time
import os
import dateutil.parser
import logging

import boto3
from boto3.dynamodb.conditions import Key
from botocore.config import Config

import langchain
from langchain.llms.bedrock import Bedrock

from chat import Chat
from fsi_agent import FSIAgent

from pypdf import PdfReader, PdfWriter

import logging
from typing import Optional

# from utils import bedrock, print_ww

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create reference to DynamoDB tables
claim_application_table_name = os.environ['USER_PENDING_ACCOUNTS_TABLE']
user_accounts_table_name = os.environ['USER_EXISTING_ACCOUNTS_TABLE']

# Instantiate boto3 clients and resources
dynamodb = boto3.resource('dynamodb', region_name=os.environ['AWS_REGION'])
s3_client = boto3.client('s3',region_name=os.environ['AWS_REGION'],config=boto3.session.Config(signature_version='s3v4',))
s3_object = boto3.resource('s3')


# --- Lex v2 request/response helpers (https://docs.aws.amazon.com/lexv2/latest/dg/lambda-response-format.html) ---
def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client

def elicit_slot(session_attributes, active_contexts, intent, slot_to_elicit, message):
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'ElicitSlot',
                'slotToElicit': slot_to_elicit
            },
            'intent': intent,
        },
        'messages': [{
            "contentType": "PlainText",
            "content": message,
        }]
    }

    return response


def confirm_intent(active_contexts, session_attributes, intent, message):
    response = {
        'sessionState': {
            'activeContexts': [active_contexts],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'ConfirmIntent'
            },
            'intent': intent
        }
    }

    return response


def close(session_attributes, active_contexts, fulfillment_state, intent, message):
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Close',
            },
            'intent': intent,
        },
        'messages': [{'contentType': 'PlainText', 'content': message}]
    }

    return response


def elicit_intent(intent_request, session_attributes, message):
    response = {
        'sessionState': {
            'dialogAction': {
                'type': 'ElicitIntent'
            },
            'sessionAttributes': session_attributes
        },
        'messages': [
            {
                'contentType': 'PlainText', 
                'content': message
            },
            {
                'contentType': 'ImageResponseCard',
                'imageResponseCard': {
                    "buttons": [
                        {
                            "text": "Claims Application",
                            "value": "Claim Application"
                        },
                        {
                            "text": "Ask GenAI",
                            "value": "What kind of questions can FSI Agent answer?" # TODO: Change this to sometihng that works correctly. 
                        }
                    ],
                    "title": "How can Aviva help you?"
                }
            }     
        ]
    }

    return response


def delegate(session_attributes, active_contexts, intent, message):
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Delegate',
            },
            'intent': intent,
        },
        'messages': [{'contentType': 'PlainText', 'content': message}]
    }

    return response


# def initial_message(intent_name):
#     response = {
#             'sessionState': {
#                 'dialogAction': {
#                     'type': 'ElicitSlot',
#                     'slotToElicit': 'UserName' if intent_name=='MakePayment' else 'PickUpCity'
#                 },
#                 'intent': {
#                     'confirmationState': 'None',
#                     'name': intent_name,
#                     'state': 'InProgress'
#                 }
#             }
#     }
    
#     return response


def build_response_card(title, subtitle, options):
    """
    Build a responseCard with a title, subtitle, and an optional set of options which should be displayed as buttons.
    """
    buttons = None
    if options is not None:
        buttons = []
        for i in range(min(5, len(options))):
            buttons.append(options[i])

    return {
        'contentType': 'ImageResponseCard',
        'imageResponseCard': {
            'title': title,
            'subTitle': subtitle,
            'buttons': buttons
        }
    }


def build_slot(intent_request, slot_to_build, slot_value):
    intent_request['sessionState']['intent']['slots'][slot_to_build] = {
        'shape': 'Scalar', 'value': 
        {
            'originalValue': slot_value, 'resolvedValues': [slot_value], 
            'interpretedValue': slot_value
        }
    }


def build_validation_result(isvalid, violated_slot, message_content):
    print("Build Validation")
    return {
        'isValid': isvalid,
        'violatedSlot': violated_slot,
        'message': message_content
    }
    

# --- Utility helper functions ---


def isvalid_date(date):
    try:
        dateutil.parser.parse(date)
        return True
    except ValueError:
        return False


def isvalid_yes_or_no(value):
    if value == 'Yes' or value == 'yes' or value == 'No' or value == 'no':
        return True
    else:
        return False


def isvalid_credit_score(credit_score):
    if int(credit_score) < 851 and int(credit_score) > 300:
        return True
    else:
        return False


def isvalid_zero_or_greater(value):
    if int(value) >= 0:
        return True
    else:
        return False


def safe_int(n):
    if n is not None:
        return int(n)
    return n


def create_presigned_url(bucket_name, object_name, expiration=600):
    # Generate a presigned URL for the S3 object
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(e)
        logging.error(e)
        return "Error"

    # The response contains the presigned URL
    return response


def try_ex(value):
    """
    Safely access Slots dictionary values.
    """
    if value is not None:
        if value['value']['resolvedValues']:
            return value['value']['interpretedValue']
        elif value['value']['originalValue']:
            return value['value']['originalValue']
        else:
            return None
    else:
        return None


# --- Intent fulfillment functions --- 


def isvalid_pin(userName, pin):
    """
    Validates the user-provided PIN using a DynamoDB table lookup.
    """
    plans_table = dynamodb.Table(user_accounts_table_name)

    try:
        # Set up the query parameters
        params = {
            'KeyConditionExpression': 'userName = :c',
            'ExpressionAttributeValues': {
                ':c': userName
            }
        }

        # Execute the query and get the result
        response = plans_table.query(**params)

        # iterate over the items returned in the response
        if len(response['Items']) > 0:
            pin_to_compare = int(response['Items'][0]['pin'])
            # check if the password in the item matches the specified password
            if pin_to_compare == int(pin):
                return True

        return False

    except Exception as e:
        print(e)
        return e


def isvalid_username(userName):
    """
    Validates the user-provided username exists in the 'user_accounts_table_name' DynamoDB table.
    """
    plans_table = dynamodb.Table(user_accounts_table_name)

    try:
        # Set up the query parameters
        params = {
            'KeyConditionExpression': 'userName = :c',
            'ExpressionAttributeValues': {
                ':c': userName
            }
        }

        # Execute the query and get the result
        response = plans_table.query(**params)

        # Check if any items were returned
        if response['Count'] != 0:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return e


def validate_pin(intent_request, slots):
    """
    Performs slot validation for username and PIN. Invoked as part of 'verify_identity' intent fulfillment.
    """
    username = try_ex(slots['UserName'])
    pin = try_ex(slots['Pin'])

    if username is not None:
        if not isvalid_username(username):
            return build_validation_result(
                False,
                'UserName',
                'Our records indicate there is no profile belonging to the username, {}. Please enter a valid username'.format(username)
            )
        session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
        session_attributes['UserName'] = username
        intent_request['sessionState']['sessionAttributes']['UserName'] = username

    else:
        return build_validation_result(
            False,
            'UserName',
            'Our records indicate there are no accounts belonging to that username. Please try again.'
        )

    if pin is not None:
        if  not isvalid_pin(username, pin):
            return build_validation_result(
                False,
                'Pin',
                'You have entered an incorrect PIN. Please try again.'.format(pin)
            )
    else:
        message = "Thank you for choosing Aviva Claims, {}. Please confirm your 4-digit PIN before we proceed.".format(username)
        return build_validation_result(
            False,
            'Pin',
            message
        )

    return {'isValid': True}


def verify_identity(intent_request):
    """
    Performs dialog management and fulfillment for username verification.
    Beyond fulfillment, the implementation for this intent demonstrates the following:
    1) Use of elicitSlot in slot validation and re-prompting.
    2) Use of sessionAttributes {UserName} to pass information that can be used to guide conversation.
    """
    slots = intent_request['sessionState']['intent']['slots']
    pin = try_ex(slots['Pin'])
    username=try_ex(slots['UserName'])

    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    intent = intent_request['sessionState']['intent']
    active_contexts = {}

    # Validate any slots which have been specified.  If any are invalid, re-elicit for their value
    validation_result = validate_pin(intent_request, intent_request['sessionState']['intent']['slots'])
    session_attributes['UserName'] = username

    if not validation_result['isValid']:
        slots = intent_request['sessionState']['intent']['slots']
        slots[validation_result['violatedSlot']] = None

        return elicit_slot(
            session_attributes,
            active_contexts,
            intent_request['sessionState']['intent'],
            validation_result['violatedSlot'],
            validation_result['message']
        )
    else:
        if confirmation_status == 'None':
            # Query DDB for user information before offering intents
            plans_table = dynamodb.Table(user_accounts_table_name)

            try:
                # Query the table using the partition key
                response = plans_table.query(
                    KeyConditionExpression=Key('userName').eq(username)
                )

                # TODO: Customize account readout based on account type
                message = ""
                items = response['Items']
                for item in items:
                    if item['planName'] == 'mortgage' or item['planName'] == 'Mortgage':
                        message = "We hear you've been involved in a crash, how might we help you?"
                    elif item['planName'] == 'Checking' or item['planName'] == 'checking':
                        message = "I see you have a Savings account with Octank Financial. Your account balance is ${:,} and your next payment \
                            amount of ${:,} is scheduled for {}.".format(item['unpaidPrincipal'], item['paymentAmount'], item['dueDate'])
                    elif item['planName'] == 'Loan' or item['planName'] == 'loan':
                            message = "I see you have a Loan account with Octank Financial. Your account balance is ${:,} and your next payment \
                            amount of ${:,} is scheduled for {}.".format(item['unpaidPrincipal'], item['paymentAmount'], item['dueDate'])
                return elicit_intent(intent_request, session_attributes, 
                    'Thank you for confirming your username and PIN, {}. {}'.format(username, message)
                    )

            except Exception as e:
                print(e)
                return e


def validate_claim_application(intent_request, slots):
    """
    Performs dialog management and fulfillment for completing a loan application.
    Beyond fulfillment, the implementation for this intent demonstrates the following:
    1) Use of elicitSlot in slot validation and re-prompting.
    2) Use of sessionAttributes to pass information that can be used to guide conversation.
    """
    username = try_ex(slots['UserName'])
    veh_involved = try_ex(slots['VehInvolved'])
    anyone_else = try_ex(slots['AnyoneElse'])
    alcohol_drugs = try_ex(slots['AlcDrugs'])
    how_many_others = try_ex(slots['AnyOthers'])


    # loan_value = try_ex(slots['LoanValue'])
    # monthly_income = try_ex(slots['MonthlyIncome'])
    # work_history = try_ex(slots['WorkHistory'])
    # credit_score = try_ex(slots['CreditScore'])
    # housing_expense = try_ex(slots['HousingExpense'])
    # debt_amount = try_ex(slots['DebtAmount'])
    # down_payment = try_ex(slots['DownPayment'])
    # coborrow = try_ex(slots['Coborrow'])
    # closing_date = try_ex(slots['ClosingDate'])

    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    intent = intent_request['sessionState']['intent']
    active_contexts = {}

    if username is not None:
        if not isvalid_username(username):
            return build_validation_result(
                False,
                'UserName',
                'Our records indicate there is no profile belonging to the username, {}. Please enter a valid username'.format(username)
            )
    else:
        try:
            session_username = intent_request['sessionState']['sessionAttributes']['UserName']
            build_slot(intent_request, 'UserName', session_username)
        except KeyError:
            return build_validation_result(
                False,
                'UserName',
                'We cannot find an account under that username. Please try again with a valid username.'
            )

    # TO DO: Mimic what is happening above with the try and then build slot. 

    if veh_involved is None: 
        reply = "What was the registration plate of the car involved?"
        return build_validation_result(False, 'VehInvolved', reply)
    else:
        try:
            # vehicle_involved = intent_request['sessionState']['intent']['slots']['VehInvolved']
            print(f"Vehicle Involved: {veh_involved}")
            build_slot(intent_request, 'VehInvolved', veh_involved)
        except KeyError:
            return build_validation_result(
                False,
                'VehInvolved',
                "What was the numberplate / registration plate of the car involved?"
            )

    if anyone_else is None: 
        reply = "Was anyone else involved in the crash? If so, was it a pedestrian, cyclist or vehicle?"
        return build_validation_result(False, 'AnyoneElse', reply) 
    else:
        try:
            # any_else = intent_request['sessionState']['sessionAttributes']['AnyoneElse']
            print(f"Anyone else?: {anyone_else}")
            build_slot(intent_request, 'AnyoneElse', anyone_else)
        except KeyError:
            return build_validation_result(
                False,
                'AnyoneElse',
                "Was anyone else involved in the crash?  If so, was it a pedestrian, cyclist or vehicle?"
            )

    if alcohol_drugs is None: 
        reply = "Were you under the influence of alcohol or drugs?"
        return build_validation_result(False, 'AlcDrugs', reply) 
    else:
        try:
            # alc_drugs = intent_request['sessionState']['sessionAttributes']['AlcDrugs']
            print(f"Booze?: {alcohol_drugs}")
            build_slot(intent_request, 'AlcDrugs', alcohol_drugs)
        except KeyError:
            return build_validation_result(
                False,
                'AlcDrugs',
                "Were you under the influence of alcohol or drugs?"
            )

    if how_many_others is None: 
        reply = "How many other people were in the vehicle, including yourself?"
        return build_validation_result(False, 'AnyOthers', reply) 
    else:
        try:
            # any_others = intent_request['sessionState']['sessionAttributes']['AnyOthers']
            print(f"How many others: {how_many_others}")
            build_slot(intent_request, 'AnyOthers', how_many_others)
        except KeyError:
            return build_validation_result(
                False,
                'AnyOthers',
                "How many other people were in the vehicle, including yourself?"
            )


    # if loan_value is not None:
    #     if not isvalid_zero_or_greater(loan_value):
    #         return build_validation_result(False, 'LoanValue', 'Please enter a value greater than $0.')
    #     else:
    #         prompt = intent_request['inputTranscript']
    #         message = invoke_fm(intent_request)
    #         reply = message + " What is your desired loan amount?"

    #         return build_validation_result(False, 'LoanValue', reply)
    # else:
    #     return build_validation_result(
    #         False,
    #         'LoanValue',
    #         "What is your desired loan amount? In other words, how much are looking to borrow? If you are unsure, please use our Loan Calculator by simply responding 'Loan Calculator.'"
    #     )

    # if monthly_income is not None:
    #     if not isvalid_zero_or_greater(monthly_income):
    #         return build_validation_result(False, 'MonthlyIncome', 'Monthly income amount must be greater than $0. Please try again.')
    #     else:
    #         prompt = intent_request['inputTranscript']
    #         message = invoke_fm(intent_request)
    #         reply = message + " What is your monthly income?"

    #         return build_validation_result(False, 'MonthlyIncome', reply)
    # else:
        # return build_validation_result(
        #     False,
        #     'MonthlyIncome',
        #     "What is your monthly income?"
        # )

    # if work_history is not None:
    #     if not isvalid_yes_or_no(work_history):
    #         return build_validation_result(False, 'WorkHistory', "I am sorry; we did not understand that. Please answer 'Yes' or 'No'")
    #     else:
    #         prompt = intent_request['inputTranscript']
    #         message = invoke_fm(intent_request)
    #         reply = message + " Do you have a two-year continuous work history (Yes/No)?"

    #         return build_validation_result(False, 'WorkHistory', reply)
    # else:
    #     return build_validation_result(
    #         False,
    #         'WorkHistory',
    #         "Do you have a two-year continuous work history (Yes/No)?"
    #     )

    # if credit_score is not None:
    #     if credit_score.isdigit():
    #         if not isvalid_credit_score(credit_score):
    #             return build_validation_result(False, 'CreditScore', 'Credit score entries must be between 300 and 850. Please enter a valid credit score.')
    #     else:
    #         prompt = intent_request['inputTranscript']
    #         message = invoke_fm(intent_request)
    #         reply = message + " What do you think your current credit score is?"

    #         return build_validation_result(False, 'CreditScore', reply)
    # else:
    #     return build_validation_result(
    #         False,
    #         'CreditScore',
    #         "What do you think your current credit score is?"
    #     )

    # if housing_expense is not None:
    #     if not isvalid_zero_or_greater(housing_expense):
    #         return build_validation_result(False, 'HousingExpense', 'Your housing expense must be a value greater than or equal to $0. Please try again.')
    #     else:
    #         prompt = intent_request['inputTranscript']
    #         message = invoke_fm(intent_request)
    #         reply = message + " How much are you currently paying for housing each month?"

    #         return build_validation_result(False, 'HousingExpense', reply)
    # else:
    #     return build_validation_result(
    #         False,
    #         'HousingExpense',
    #         "How much are you currently paying for housing each month?"
    #     )

    # if debt_amount is not None:
    #     if not isvalid_zero_or_greater(debt_amount):
    #         return build_validation_result(False, 'DebtAmount', 'Your debt amount must be a value greater than or equal to $0. Please try again.')
    #     else:
    #         prompt = intent_request['inputTranscript']
    #         message = invoke_fm(intent_request)
    #         reply = message + " What is your estimated credit card or student loan debt?"

    #         return build_validation_result(False, 'DebtAmount', reply)
    # else:
    #     return build_validation_result(
    #         False,
    #         'DebtAmount',
    #         "What is your estimated credit card or student loan debt? Please enter '0' if none."
    #     )

    # if down_payment is not None:
    #     if not isvalid_zero_or_greater(down_payment):
    #         return build_validation_result(False, 'DownPayment', 'Your estimate down payment must be a value greater than or equal to $0. Please try again.')
    #     else:
    #         prompt = intent_request['inputTranscript']
    #         message = invoke_fm(intent_request)
    #         reply = message + " What do you have saved for a down payment?"

    #         return build_validation_result(False, 'DownPayment', reply)
    # else:
    #     return build_validation_result(
    #         False,
    #         'DownPayment',
    #         "What do you have saved for a down payment?"
    #     )

    # if coborrow is not None:
    #     if not isvalid_yes_or_no(coborrow):
    #         return build_validation_result(False, 'Coborrow', "I am sorry; we did not understand that. Please answer 'Yes' or 'No'")
    #     else:
    #         prompt = intent_request['inputTranscript']
    #         message = invoke_fm(intent_request)
    #         reply = message + " Do you have a co-borrower (Yes/No)?"

    #         return build_validation_result(False, 'Coborrow', reply)
    # else:
    #     return build_validation_result(
    #         False,
    #         'Coborrow',
    #         "Do you have a co-borrower (Yes/No)?"
    #     )

    # if closing_date is not None:
    #     if not isvalid_date(closing_date):
    #         return build_validation_result(False, 'ClosingDate', 'I did not understand your closing date.  When would you like to close?')
    #     if datetime.datetime.strptime(closing_date, '%Y-%m-%d').date() <= datetime.date.today():
    #         return build_validation_result(False, 'ClosingDate', 'Closing dates must be scheduled at least one day in advance.  Please try a different date.')
    #     else:
    #         prompt = intent_request['inputTranscript']
    #         message = invoke_fm(intent_request)
    #         reply = message + " When are you looking to close?"

    #         return build_validation_result(False, 'ClosingDate', reply)        
    # else:
        # return build_validation_result(
        #     False,
        #     'ClosingDate',
        #     'When are you looking to close?'
        # )

    return {'isValid': True}


def claim_application(intent_request):
    """
    Performs dialog management and fulfillment for booking a car.
    Beyond fulfillment, the implementation for this intent demonstrates the following:
    1) Use of elicitSlot in slot validation and re-prompting
    2) Use of sessionAttributes to pass information that can be used to guide conversation
    """
    slots = intent_request['sessionState']['intent']['slots']

    username = try_ex(slots['UserName'])
    veh_involved = try_ex(slots['VehInvolved'])
    anyone_else = try_ex(slots['AnyoneElse'])
    alcohol_drugs = try_ex(slots['AlcDrugs'])
    how_many_others = try_ex(slots['AnyOthers'])

    # loan_value = try_ex(slots['LoanValue'])
    # monthly_income = try_ex(slots['MonthlyIncome'])
    # work_history = try_ex(slots['WorkHistory'])
    # credit_score = try_ex(slots['CreditScore'])
    # housing_expense = try_ex(slots['HousingExpense'])
    # debt_amount = try_ex(slots['DebtAmount'])
    # down_payment = try_ex(slots['DownPayment'])
    # coborrow = try_ex(slots['Coborrow'])
    # closing_date = try_ex(slots['ClosingDate'])

    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    intent = intent_request['sessionState']['intent']
    active_contexts = {}

    print(f"Confirmation Status: {confirmation_status}")
    print(f"Session Attributes: {session_attributes}")
    print(f"Intent: {intent}")
    
    if intent_request['invocationSource'] == 'DialogCodeHook':
        # Validate any slots which have been specified.  If any are invalid, re-elicit for their value

        # TODO: Bring this back into the mix somehow after validating the values that I pass in. 
        validation_result = validate_claim_application(intent_request, intent_request['sessionState']['intent']['slots'])

        print(f"Validation result is: {validation_result}")

        if not validation_result['isValid']:
            if validation_result['violatedSlot'] == 'CreditScore' and confirmation_status == 'Denied':
                print("Invalid credit score")
                validation_result['violatedSlot'] = 'UserName'
                intent['slots'] = {}
            slots[validation_result['violatedSlot']] = None

            print("Now elicting slot!!!")
            response_going_back = elicit_slot(
                session_attributes,
                active_contexts,
                intent,
                validation_result['violatedSlot'],
                validation_result['message']
            )  
            print(f"Response going back: {response_going_back}")

            return response_going_back

    if username:
        # application = {
        #     'ApplicationType': 'Claim',
        #     'Username': username,
        #     'VehInv': loan_value,
        #     'MonthlyIncome': monthly_income,
        #     'CreditScore': credit_score,
        #     'DownPayment': down_payment
        # }

        # # Convert the JSON document to a string
        # application_string = json.dumps(application)

        # # Write the JSON document to DynamoDB
        # claim_application_table = dynamodb.Table(claim_application_table_name)

        # response = claim_application_table.put_item(
        #     Item={
        #         'userId': username,
        #         'document': application_string
        #     }
        # )

        # Determine if the intent (and current slot settings) has been denied.  The messaging will be different
        # if the user is denying a reservation he initiated or an auto-populated suggestion.
        # if confirmation_status == 'Denied':
        #     return delegate(session_attributes, active_contexts, intent, 'Confirm hotel reservation')

        # if confirmation_status == 'None':
        #     return delegate(session_attributes, active_contexts, intent, 'Confirm hotel reservation')

        confirmation_status = 'Confirmed'
        if confirmation_status == 'Confirmed':
            intent['confirmationState'] = "Confirmed"
            intent['state'] = "Fulfilled"

            s3_client.download_file('avivabedrockapp-719514420056', 'agent/assets/Aviva-Claim-Application.pdf', '/tmp/Aviva-Claim-Application.pdf')

            reader = PdfReader('/tmp/Aviva-Claim-Application.pdf')
            writer = PdfWriter()

            page = reader.pages[0]
            fields = reader.get_fields()

            writer.append(reader)

            firstname, lastname = username.split(' ', 1)
            writer.update_page_form_field_values(
                writer.pages[0], {
                    'fullName34[first]': firstname,
                    'fullName34[last]': lastname,
                    'monthlyNet': veh_involved,
                    'creditScore': anyone_else,
                    'requestedLoan': alcohol_drugs,
                    'downPayment': how_many_others
                }
            )

            with open('/tmp/Aviva-Claim-Application.pdf', "wb") as output_stream:
                writer.write(output_stream)
                
            s3_client.upload_file('/tmp/Aviva-Claim-Application.pdf', 'avivabedrockapp-719514420056', 'agent/assets/Aviva-Claim-Application-Completed.pdf')

            # Create loan application doc in S3
        URLs=[]

        # create_presigned_url(bucket_name, object_name, expiration=600):
        URLs.append(create_presigned_url('avivabedrockapp-719514420056','agent/assets/Aviva-Claim-Application-Completed.pdf',3600))
        
        claim_app = 'Your claim application is nearly complete! Please follow the link for the last few bits of information: ' + URLs[0]

        print("But does it elicit intent? ")
        return elicit_intent(
            intent_request,
            session_attributes,
            claim_app
        )


# def loan_calculator(intent_request):
#     """
#     Performs dialog management and fulfillment for calculating loan details.
#     This is an empty function framework intended for the user to develope their own intent fulfillment functions.
#     """
#     session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}

#     # def elicit_intent(intent_request, session_attributes, message)
#     return elicit_intent(
#         intent_request,
#         session_attributes,
#         'This is where you would implement LoanCalculator intent fulfillment.'
#     )


def invoke_fm(intent_request):
    """
    Invokes Foundational Model endpoint hosted on Amazon Bedrock and parses the response.
    """
    prompt = intent_request['inputTranscript']
    chat = Chat(prompt)

    # bedrock_model_id = "anthropic.claude-instant-v1"

    bedrock_runtime = get_bedrock_client(
                    assumed_role= "arn:aws:iam::195364414018:role/Crossaccountbedrock", # os.environ.get("BEDROCK_ASSUME_ROLE", None),
                    region="us-east-1"
                )

    llm = Bedrock(
        model_id="anthropic.claude-instant-v1", 
        client=bedrock_runtime
    )
    llm.model_kwargs = {'max_tokens_to_sample': 200}
    lex_agent = FSIAgent(llm, chat.memory)

    # langchain.debug = True

    try:
        message = lex_agent.run(input=prompt)
    except ValueError as e:
        print(f"the message coming back is: {message}")
        message = str(e)
        if not message.startswith("Could not parse LLM output: `"):
            raise e
        message = message.removeprefix("Could not parse LLM output: `").removesuffix("`")
        return message

    output = message['output']

    return output


def genai_intent(intent_request):
    """
    Performs dialog management and fulfillment for user utterances that do not match defined intents (i.e., FallbackIntent).
    Sends user utterance to Foundational Model endpoint via 'invoke_fm' function.
    """
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    
    if intent_request['invocationSource'] == 'DialogCodeHook':
        output = invoke_fm(intent_request)

        logger.info(f"genai_intent output: {output}")
        return elicit_intent(intent_request, session_attributes, output)


# --- Intents ---


def dispatch(intent_request):
    """
    Routes the incoming request based on intent.
    """
    slots = intent_request['sessionState']['intent']['slots']
    username = slots['UserName'] if 'UserName' in slots else None
    intent_name = intent_request['sessionState']['intent']['name']

    if intent_name == 'VerifyIdentity':
        # return verify_identity(intent_request)
        intent_name = 'LoanApplication'

    if intent_name == 'LoanApplication':
        print(f"picking up the intent name: {intent_name}")
        return claim_application(intent_request)
    # elif intent_name == 'LoanCalculator':
    #     return loan_calculator(intent_request)
    else:
        return genai_intent(intent_request)

    raise Exception('Intent with name ' + intent_name + ' not supported')
        

# --- Main handler ---


def handler(event, context):
    """
    Invoked when the user provides an utterance that maps to a Lex bot intent.
    The JSON body of the user request is provided in the event slot.
    """

    os.environ['TZ'] = 'America/New_York'
    time.tzset()
    print(f"THE EVENT IS: {event}")
    return dispatch(event)