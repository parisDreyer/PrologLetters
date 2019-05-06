import sys
sys.path.append('./py/lstm_character_level_chatbot')
sys.path.append('./py/lstm_word_level_chatbot')
sys.path.append('./py/lstm_sentence_level_chatbot')
from pyswip import Prolog   # https://pypi.org/project/pyswip/
import lstm_character_level_chatbot
import lstm_word_level_chatbot
import lstm_sentence_level_chatbot
import string
import util
import bot_output


class ChatPy:
    def __init__(self):
        self.user_input = ""
        self.prolog = Prolog()
        self.prolog.consult("./ask.pl")
        self.explanation = "Explanation"
        self.character_bot = bot_output.BotOutput(lstm_character_level_chatbot)
        self.word_bot = bot_output.BotOutput(lstm_word_level_chatbot)
        self.sentence_bot = bot_output.BotOutput(lstm_sentence_level_chatbot)

    def main(self):
        while self.user_input.upper() != "EXIT":
            self.user_input = raw_input("exit, or:> ")
            print(self.consider_input())
            print("\n______________________________________________")


    def consider_input(self):
        pre_prepped_input = str(self.user_input).translate(None, string.punctuation).lower()
        prepped_input = "', '".join(pre_prepped_input.split(' '))
        if len(prepped_input) > 0:
            prolog_ask = "ask([\'" + prepped_input + "\'], " + self.explanation + ")"
            return self.prolog_output(prolog_ask) + self.bot_output(pre_prepped_input)
        return ""

    def prolog_output(self, question):
        response = ""
        for dictionary_object in self.prolog.query(question):
            response += " ".join(dictionary_object.get(self.explanation, [""])) + " "
        return response

    def bot_output(self, input):
        self.character_bot.append_user_input(input)
        self.word_bot.append_user_input(input)
        self.sentence_bot.append_user_input(input)
        self.character_bot.generate_response()
        self.word_bot.generate_response()
        self.sentence_bot.generate_response()
        return self.character_bot.report_output() + self.word_bot.report_output() + self.sentence_bot.report_output()

         


    
