from main import ToxicLanguage

val = ToxicLanguage(
        use_local=False)



if __name__ == "__main__":
    result = val.validate("Fuck you bitch I love you", metadata={})
    print(result)
    result = val.validate("shit, I love you so much girl. You are my favorite in the world. But fuck you.", metadata={})
    print(result)

    result = val.validate("You are my favorite in the world", metadata={})
    print(result)