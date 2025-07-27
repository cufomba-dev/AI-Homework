from django.core.management.base import BaseCommand
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

class Command(BaseCommand):
    help = "Starts a chatbot terminal client"

    def handle(self, *args, **kwargs):
        chatbot = ChatBot('TerminalBot')
        trainer = ChatterBotCorpusTrainer(chatbot)
        trainer.train("chatterbot.corpus.english")

        print("\n" + "="*50)
        print("Welcome to Terminal Bot Chat!")
        print("Type 'quit' or 'exit' to end the conversation")
        print("="*50 + "\n")

        while True:
            try:
                user_input = input("user: ")
                if user_input.lower() == 'exit':
                    print("Goodbye! Thanks for chatting with me.")
                    break

                response = chatbot.get_response(user_input)
                print(f"bot: {response}")

            except (KeyboardInterrupt, EOFError):
                print("Exiting...")
                break