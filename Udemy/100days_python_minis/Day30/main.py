def day30s1():

    try:
        file = open("scratchpad.txt")
        my_dict = {"abc":123}
    except FileNotFoundError:
        file = open("scratchpad.txt","w")
        file.write("weeeee")
    except KeyError as error_var:
        print(f"keyerror output as variable is {error_var}")
    else:
        print("this line runs because the try was successful")
        file.close()

day30s1()