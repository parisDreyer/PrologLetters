import util
from threading import Thread

class BotOutput:
    def __init__(self, bot):
        self.bot = bot
        self.input_store = []
        self.output_store = []
        self.is_fetching_output = False

    def generate_response(self):
        if ((self.is_fetching_output == False) and (len(self.input_store) > 0)):
            input = self.input_store.pop()
            self.is_fetching_output = True
            output_thread = Thread(target=self.append_bot_output, args=(input,))
            output_thread.start()
            self.append_bot_output(input)


    def append_bot_output(self, input):
        self.output_store.append(self.bot.response(input))
        self.is_fetching_output = False
        print(":>")

    def report_output(self):
        bot_output = ""
        if len(self.output_store) > 0:
            bot_output = self.output_store.pop()
        if type(bot_output) != type(None) and len(bot_output) > 0:
            return " {}".format(util.format_bot_output(bot_output))
        return ""

    def append_user_input(self, input):
        self.input_store.append(input)
