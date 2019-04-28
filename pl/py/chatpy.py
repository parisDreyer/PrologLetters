from pyswip import Prolog   # https://pypi.org/project/pyswip/
import lstm_character_level_chatbot
import string

class ChatPy:
    def __init__(self):
        self.user_input = ""
        self.temp_user_input_store = []
        self.prolog = Prolog()
        self.prolog.consult("./ask.pl")

    def main(self):
        while self.user_input.upper() != "EXIT":
            self.user_input = raw_input("exit, or:> ")
            output = self.consider_input(self.prep_for_prolog_ask())
            print(output)
            print("\n______________________________________________")

    def prep_for_prolog_ask(self):
        prepped = "', '".join(str(self.user_input).translate(None, string.punctuation).lower().split(' '))
        if len(prepped) > 0:
            prepped = "ask([\'{}\'], Explanation)".format(prepped)
        return prepped

    def consider_input(self, prolog_ask):
        answer = ""
        if len(prolog_ask) > 0:
            answer = self.decide_output(prolog_ask)
        else:
            answer = "didn\'t find input for: {}\n".format(self.user_input)
        return answer

    def decide_output(self, prolog_ask):
        generated_output_count = 0
        response = ""
        for dictionary_object in self.prolog.query(prolog_ask):
            # for entry in dictionary_object.itervalues():
            generated_output_count += 1
            response += str(dictionary_object)
        joined_input_store = " ".join(self.temp_user_input_store)
        if generated_output_count == 0 and len(joined_input_store) > lstm_character_level_chatbot.maxlen:
            print("in lstm generator:\n")
            response = lstm_character_level_chatbot.response(joined_input_store)
            self.temp_user_input_store = []             # reset temp_user_input_store
        else:
            print("incrementing user input")
            self.temp_user_input_store.append(self.user_input)
        return response
