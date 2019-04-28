import sys

sys.path.append('./py/lstm_chatbot')
import lstm_chatbot
from pyswip import Prolog #https://pypi.org/project/pyswip/
prolog = Prolog()
prolog.consult("ask.pl")
user_input = ""
while user_input.upper() != "EXIT":
    user_input = raw_input("exit, or:> ")
    modified_user_input = "', '".join(str(user_input).split(' '))
    generated_output_count = 0
    if len(user_input) > 0:
        results = prolog.query("ask([\'{}\']).".format(modified_user_input))
        for res in results:
            generated_output_count += 1
            print(res)
        if generated_output_count == 0:
            response = lstm_chatbot.lstm_chatbot_response(user_input)
            print(response)
    else:
        print("didn\'t find input for: {}".format(user_input))
