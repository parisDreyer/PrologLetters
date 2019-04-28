from pyswip import Prolog   # https://pypi.org/project/pyswip/
import lstm_character_level_chatbot
import string

class ChatPy:
    def __init__(self):
        self.user_input = ""
        self.temp_user_input_store = []
        self.prolog = Prolog()
        self.prolog.consult("./ask.pl")
        self.explanation = "Explanation"

    def main(self):
        while self.user_input.upper() != "EXIT":
            self.user_input = raw_input("exit, or:> ")
            print(self.consider_input(self.prep_for_prolog_ask()))
            print("\n______________________________________________")

    def prep_for_prolog_ask(self):
        prepped = "', '".join(str(self.user_input).translate(None, string.punctuation).lower().split(' '))
        if len(prepped) > 0:
            prepped = "ask([\'" + prepped + "\'], " + self.explanation + ")"
        return prepped

    def consider_input(self, prolog_ask):
        answer = ""
        if len(prolog_ask) > 0:
            answer = self.decide_output(prolog_ask)
        else:
            answer = "didn\'t find input for: {}\n".format(self.user_input)
        return answer

    def decide_output(self, prolog_ask):
        response = ""
        for dictionary_object in self.prolog.query(prolog_ask):
            response += " ".join(dictionary_object.get(self.explanation, [""])) + " "
        joined_input_store = " ".join(self.temp_user_input_store)
        if len(response) == 0 or len(joined_input_store) > lstm_character_level_chatbot.maxlen:
            response = lstm_character_level_chatbot.response(joined_input_store)
            self.temp_user_input_store = []             # reset temp_user_input_store
        else:
            self.temp_user_input_store.append(self.user_input)
            response = lstm_character_level_chatbot.response(response)
        return response
