import random
from IPython.display import clear_output
import time
def multiple_choice(optionsDict, questionTemplate, action, num2win):
    tryagain = True
    # clear text from output and print the next question.
    num_correct = 0

    while tryagain:
        clear_output(wait=True)

        answers = list(set(optionsDict.values()))
        var = random.choice(list(optionsDict.keys()))
        type_var = optionsDict[var]
        
        question = questionTemplate.replace('XXX', var)
        print(question)

        options = random.sample(answers, 5)
        options.append(type_var)
        random.shuffle(options)
        options = list(set(options))

        for i in range(len(options)):
            print(f"{i}: {options[i]}")
        
        answer = None
        while not answer:
            answer = input(f"Choose the correct answer (0-{len(options)-1}). If you want to quit, type 'q': ")
            if answer == 'q':
                print(f"You answered {num_correct} questions correctly. Goodbye!") 
                tryagain = False
                return
            if not answer.isdigit() or int(answer) not in range(len(options)):
                print(f"Please enter a number between 0 and {len(options)-1}.")
                answer = None
        if options[int(answer)] == type_var:
            num_correct += 1
            print(f"Yay - good job! You have {num_correct} correct answers. Get {num2win} correct answers to win the game.")
            if num_correct >= num2win:
                print('\nYou won the game!\n')
                print(action)
                print('\nPlease keep going if you want to practice more.')

        else:
            print(f"I'm sorry, the correct answer is {type_var}")

        time.sleep(1)
        tryagain = input("Do you want to try again? ([y]/n): ") != 'n'
    print("Goodbye!")

def multiple_choice_quiz(quizName):
    match quizName.lower():
        case "types":
            optionsDict = {'A = 1':'int', 'B = 2.0':'float', 'C = "3"':'string', 'D = True':'bool', 'E = [1,2,3]':'list',  'F = (1,2,3)':'tuple', 'G = {1,2,3}':'set', 'H = {"a":1,"b":2,"c":3}':'dict', 'I = None':'NoneType', 'J = 1+2j':'complex', 
                    'K = 1.3':'float', 'L = 5-2j':'complex', 'M = {1:2, 3:4}':'dict', "N = {'a','b','c'}":'set', 'O = "hello"':'string', "P = ['x','y','z']":'list', "Q = ('q','r','s')":'tuple', 'R = False':'bool', 'S = None':'NoneType'}
            question = "What is the type of the following variable: XXX ?"
            action = "You got 5 right - place your left hand on your head."
            num2win = 5
        case "slicing":
            question = "How would you index or slice the the list 'x' to get XXX ?"
            optionsDict = {'the first element':'x[0]', 
                        'the last element':'x[-1]', 
                        'a list with the first 3 elements':'x[:3]', 
                        'a list with the last 3 elements':'x[-3:]', 
                        'a list with the second to the fourth element':'x[1:4]', 
                        'a list with the second to the last element':'x[1:-1]', 
                        'a list with the first element and the last element':'[x[0], x[-1]]', 
                        'a list with the last element and the second to the last element':'x[-1], x[-2]', 
                        'a list with the first 3 elements and the last 3 elements':'[x[:3], x[-3:]]', 
                        'a list with every third element starting with the first element':'x[::3]',
                        'a list with every third element starting with the second element':'x[1::3]',
                        'a reversed list with every third element starting with the last element':'x[-1::-3]',
                        'a list of every string in the list that has the letter "a"':'[i for i in x if "a" in i]',
                        'a list of every string in the list that has the letter "a" in the second position':'[i for i in x if len(i)>1 and i[1]=="a"]'}
            action = "You got 8 right - touch your nose with your right hand."
            num2win = 8
        case "list_operations":
            question = "Which command would you use to perform the following operation on the lists x, y and z:\n XXX ?"
            optionsDict = {'concatenate the lists to form a longer list':'x + y + z',
                        'repeat the list x 5 times':'x*5',
                        'remove repeated elements from the list x':'list(set(x))',
                        'sort the list x':'sorted(x)',
                        'reverse the list x':'x[::-1]',
                        'add the element 5 to the end of the list x':'x.append(5)',
                        'remove the first element from the list x':'x.pop(0)',
                        'sum the elements of the list x':'sum(x)',
                        "find the first index of the element 'abc' in the list x":'x.index("abc")',
                        'count the number of times the element 5 appears in the list x':'x.count(5)',
                        'create a new list with the summed elements of the lists x and y':'[i+j for i,j in zip(x,y)]',
                        'create a tuple with the elements of the list x':'tuple(x)',
                        'create a set with the elements of the list x':'set(x)',
                        'create a dictionary with the elements of the list x':'dict(enumerate(x))',
                        'find the intersection of the lists x and y':'list(set(x) & set(y))',
                        'find the union of the lists x and y':'list(set(x) | set(y))',
                        'find the difference of the lists x and y':'list(set(x) - set(y))',
                        'find the symmetric difference of the lists x and y':'list(set(x) ^ set(y))',
                        'check if the element 5 is in the list x':'if 5 in x',
                        'check if the element 5 is not in the list x':'if 5 not in x',
                        'check if the list x is empty':'if not x',
                        'check if the list x is not empty':'if x',
                        'check if the list x is equal to the list y':'if x==y',
                        'check if the list x is not equal to the list y':'if x!=y',
                        'check if maxiumum element of the list x is greater than 5':'if max(x)>5',
                        'check if minimum element of the list x is less than 5':'if min(x)<5',
                        'check if all elements of the list x are greater than 5':'if all(i>5 for i in x)',
                        'check if any element of the list x is greater than 5':'if any(i>5 for i in x)',
                        'check if all elements of the list x are strings':'if all(isinstance(i, str) for i in x)',
                        'check if any element of the list x is a string':'if any(isinstance(i, str) for i in x)',
                        'check if all elements of the list x are unique':'if len(x)==len(set(x))'}  
            action = "You got 12 right - look up and smile."
            num2win = 12
            
        case "dictionary_operations":
            question = "Which command would you use to perform the following operation on the lists X, Y, and Z dictionaries a, b and c:\n XXX ?"  
            optionsDict = {'create a new dictionary called using X as the keys and Y as the values':'a = dict(zip(X,Y))',
                        'extract the value of the key "abc" from the dictionary a':'a["abc"]',
                        'add the key "abc" with the value 5 to the dictionary a':'a["abc"] = 5',
                        'remove the key "abc" from the dictionary a':'a.pop("abc")',
                        'remove all elements from the dictionary a':'a.clear()',
                        'create a subset of the dictionary a with the keys in the list X':'{k:a[k] for k in X}',
                        'create a subset of the dictionary a with the values in the list Y':'{k:v for k,v in a.items() if v in Y}',
                        'create a subset of the dictionary a with the keys that contain the letter "a"':'{k:v for k,v in a.items() if "a" in k}',
                        'create a subset of the dictionary a with the values that are greater than 5':'{k:v for k,v in a.items() if v>5}',
                        'get a list of all the keys in the dictionary a':'list(a.keys())',
                        'get a list of all the values in the dictionary a':'list(a.values())',
                        'get a list of all unique values in the dictionary a':'list(set(a.values()))',
                        'get a list of all key-value pairs in the dictionary a':'list(a.items())',
                        'check if the key "abc" is in the dictionary a':'if "abc" in a',
                        'check if the key "abc" is not in the dictionary a':'if "abc" not in a',
                        'check if the value 5 is in the dictionary a':'if 5 in a.values()',
                        'check if the value 5 is not in the dictionary a':'if 5 not in a.values()',
                        'check if the dictionary a is empty':'if not a',
                        'check if the dictionary a is not empty':'if a',
                        'check if the dictionary a is equal to the dictionary b':'if a==b',
                        'check if the dictionary a is not equal to the dictionary b':'if a!=b',
                        'check if the dictionary a has the key "abc"':'if "abc" in a.keys()',
                        'check how many key-value pairs are in the dictionary a':'len(a)'}
            num2win = 10            
            action = f"You got 10 right - cough three times."

                        
        case "string_operations":
            question = "Which command would you use to perform the following operation on the strings X and/or Y:\n XXX ?"
            optionsDict = {'get the first character of the string X':'X[0]',
                        'get the last character of the string X':'X[-1]',
                        'get the first 3 characters of the string X':'X[:3]',
                        'get the last 3 characters of the string X':'X[-3:]',
                        'split the string X into a list of words separaed by spaces':'X.split()',
                        'split the string X into a list of phrases separated by commas':'X.split(",")',
                        'split the string X into a list of characters':'list(X)',
                        'join the a list Y containing words into a single string separated by spaces':'" ".join(Z)',
                        'join the a list Y containing words into a single string separated by commas':'",".join(Z)',
                        'concatenate the strings X and Y':'X + Y',
                        'count the number of times the letter "a" appears in the string X':'X.count("a")',
                        'change the string X to uppercase':'X.upper()',
                        'change the string X to lowercase':'X.lower()',
                        'change an integer x to a string':'str(x)',
                        'change a string x to an integer':'int(x)',
                        'change a string x to a float':'float(x)',
                        'count how many times the letters "aa" appear in the string X':'sum(1 for i in range(len(X)-1) if X[i:i+2]=="aa")',
                        'replace the letter "a" with the letter "b" in the string X':'X.replace("a","b")',
                        'check if the string X is equal to the string Y':'if X==Y',
                        'check if the string X is not equal to the string Y':'if X!=Y',
                        'check if the string X is in the string Y':'if X in Y',
                        'check if the string X is not in the string Y':'if X not in Y',
                        'check if the string X is empty':'if not X',
                        'check if the string X is not empty':'if X',
                        'check if the string X is a palindrome':'if X==X[::-1]',
                        'flip the order of the words in the string X':'" ".join(X.split()[::-1])',
                        'get a list of all unique words in the string X':'list(set(X.split()))',
                        'get the length of the string X':'len(X)',
                        'get the length of the string X without counting spaces':'len(X.replace(" ",""))',
                        'create a random string of length 10':'"".join(random.choices(string.ascii_letters + string.digits, k=10))',
                        'create a random string of length 10 with only letters':'"".join(random.choices(string.ascii_letters, k=10))'}
            action = "You got 15 right - knock twice on the table."
            num2win = 15

        case 'math_operations':
            question = "Which command would you use to perform the following operation on the numbers X and Y:\n XXX ?"
            optionsDict = {'add the numbers X and Y':'X + Y',
                        'subtract the number Y from X':'X - Y',
                        'multiply the numbers X and Y':'X * Y',
                        'divide the number X by Y':'X / Y',
                        'raise the number X to the power of Y':'X ** Y',
                        'find the remainder of X divided by Y':'X % Y',
                        'find the integer division of X divided by Y':'X // Y',
                        'find the square root of X':'X ** 0.5',
                        'find the absolute value of X':'abs(X)',
                        'find the maximum of X and Y':'max(X,Y)',
                        'find the minimum of X and Y':'min(X,Y)',
                        'round the number X to the nearest integer':'round(X)',
                        'round the number X to the nearest tenth':'round(X,1)',
                        'round the number X to the nearest hundredth':'round(X,2)',
                        'round the number X to the nearest thousandth':'round(X,3)',
                        'check if the number X is equal to the number Y':'if X==Y',
                        'check if the number X is not equal to the number Y':'if X!=Y',
                        'check if the number X is greater than the number Y':'if X>Y',
                        'check if the number X is less than the number Y':'if X<Y',
                        'check if the number X is greater than or equal to the number Y':'if X>=Y',
                        'check if the number X is less than or equal to the number Y':'if X<=Y',
                        'check if the number X is positive':'if X>0',
                        'check if the number X is negative':'if X<0',
                        'check if the number X is even':'if X%2==0',
                        'check if the number X is odd':'if X%2!=0',
                        'check if the number X is a prime number':'if all(X%i!=0 for i in range(2,X))',
                        'check if the number X is a multiple of Y':'if X%Y==0',
                        'check if the number X is a power of Y':'if X == Y ** round(math.log(X, Y))'}
            action = f"You got 10 right - stand up and then sit down."
            num2win = 10
        case 'string_formatting':
            question = "Which command would you use to perform the following operation on the strings X and Y:\n XXX ?"
            optionsDict = {'concatenate the strings X and Y':'X + Y',
                        'concatenate the strings X and Y with a space in between':'X + " " + Y',
                        'complete a string to include the value of the variable X':'f"Value of X is {X}"',
                        'complete a string to include the value of the variable X with 2 decimal places':'f"Value of X is {X:.2f}"',
                        'complete a string to include the values and sum of the variables X and Y':'f"Values of X and Y are {X} and {Y} and their sum is {X+Y}"',
                        'create a string will all the numbers in the list X on a single line':'" ".join(map(str,X))',
                        'create a string will all the numbers in the list X on separate lines':'"\\n".join(map(str,X))',
                        'create a string with the numbers in the list X separated by commas':'",".join(map(str,X))',
                        'create a string with the numbers in the list X separated by tabs':'"\\t".join(map(str,X))',
                        'align the string X to the left in a field of width 10':'X.ljust(10)',
                        'align the string X to the right in a field of width 10':'X.rjust(10)',
                        'center the string X in a field of width 10':'X.center(10)',
                        'fill the string X with dashes to make it 10 characters long':'X.ljust(10,"-")',
                        'convert a string X to uppercase':'X.upper()',
                        'convert a string X to lowercase':'X.lower()',
                        'capitalize the first letter of the string X':'X.capitalize()',
                        'capitalize the first letter after a space in the string X':'" ".join(i.capitalize() for i in X.split())'}
            action = "You got 10 right - touch your left ear."
            num2win = 10

        case 'importing_and_using_modules':
            question = "Which command would you use to perform the following operation on the module X:\n XXX ?"
            optionsDict = {'import the module X':'import X',
                        'import the module X and give it the alias Y':'import X as Y',
                        'import a specific function Z from the module X':'from X import Z',
                        'import all functions from the module X':'from X import *',
                        'reload the module X':'import importlib; importlib.reload(X)',
                        'check the version of the module X':'X.__version__',
                        'check the documentation of the module X':'help(X)',
                        'list the functions in the module X':'dir(X)',
                        'list the functions in the module X that start with the letter "a"':'[i for i in dir(X) if i.startswith("a")]',
                        'list the functions in the module X that contain the letter "a"':'[i for i in dir(X) if "a" in i]',
                        'list the modules in the current environment':'!pip list',
                        'list the modules in the current environment that start with the letter "a"':'!pip list | grep "^a"',
                        'list the modules in the current environment that contain the letter "a"':'!pip list | grep "a"',
                        'check if the module X is installed':'!pip show X',
                        'install the module X':'!pip install X',
                        'uninstall the module X':'!pip uninstall X',
                        'update the module X':'!pip install --upgrade X',
                        'show the current working directory':'!pwd or %pwd or os.getcwd()',
                        'change the current working directory to X':'!cd X or %cd X or os.chdir(X)',
                        'list the files in the current working directory':'!ls or %ls or os.listdir()',
                        'create a new directory called X':'!mkdir X or os.mkdir(X)',
                        'show the search path for modules':'sys.path',
                        'add a directory to the search path for modules':'sys.path.append(X)',
                        'show the user environment variables':'os.environ',
                        'show the user home directory':'os.path.expanduser("~")',
                        'show the user name':'os.getlogin()',
                        'shortcut to the home directory':'~',
                        'shortcut to the current directory':'.',
                        'shortcut to the parent directory':'..'}
            action = "You got 8 right - touch your palm to your forehead."
            num2win = 8
        case 'command_line_tools':
            question = "Which command would you use to perform the following operation on the command line:\n XXX ?"
            optionsDict = {'list the files in the current directory without details':'ls',
                        'list the files in the current directory with details':'ls -l',
                        'list the files in the current directory with details and hidden files':'ls -la',
                        'change the current directory to X':'cd X',
                        'change the current directory to the parent directory':'cd ..',
                        'change the current directory to the home directory':'cd ~',
                        'create a new directory called X':'mkdir X',
                        'remove the empty directory X':'rmdir X',
                        'remove the directory X and all its contents':'rm -r X',
                        'copy the file X to the directory Y':'cp X Y',
                        'move the file X to the directory Y':'mv X Y',
                        'rename the file X to Y':'mv X Y',
                        'remove the file X':'rm X',
                        'remove the file X without confirmation':'rm -f X',
                        'print the contents of the file X':'cat X',
                        'print the first 10 lines of the file X':'head X',
                        'print the last 10 lines of the file X':'tail X',
                        'print the first 5 lines of the file X that contain the word Y':'grep Y X | head -5',
                        'print the last 5 lines of the file X that contain the word Y':'grep Y X | tail -5',
                        'print the number of lines in the file X':'wc -l X',
                        'print the number of words in the file X':'wc -w X',
                        'print the number of characters in the file X':'wc -c X',
                        'print the number of lines, words, and characters in the file X':'wc X',
                        'print a list of recent commands':'history',
                        'repeat the last command':'!!',
                        'repeat the last command that started with the letter X':'!X',
                        'get detailed information about the command X':'man X',
                        'find the location of the command X':'which X',
                        'print the current working directory':'pwd',
                        'print the current user name':'whoami',
                        'print the current date and time':'date'}
            action = "You got 10 right - touch your left ear."
            num2win = 10

        case 'file_operations':
            question = "Which command would you use to perform the following operation on the file X:\n XXX ?"
            optionsDict = {'open the file X in read mode':'open(X, "r")',
                        'open the file X in write mode':'open(X, "w")',
                        'open the file X in append mode':'open(X, "a")',
                        'open the file X in read and write mode':'open(X, "r+")',
                        'close the file X':'X.close()',
                        'read the entire contents of the file X':'X.read()',
                        'read the first line of the file X':'X.readline()',
                        'read all lines of the file X into a list':'X.readlines()',
                        'write the string Y to the file X':'X.write(Y)',
                        'write the list Y to the file X':'X.writelines(Y)',
                        'check if the file X is closed':'X.closed',
                        'check if the file X is readable':'X.readable()',
                        'check if the file X is writable':'X.writable()',
                        'check if the file X is seekable':'X.seekable()',
                        'check if the file X is at the end':'X.seek(0,2)',
                        'move the file X cursor to the beginning':'X.seek(0)',
                        'move the file X cursor to the end':'X.seek(0,2)',
                        'move the file X cursor to the 5th byte':'X.seek(5)',
                        'move the file X cursor to the 5th line':'X.seek(0); [X.readline() for i in range(5)]',
                        'move the file X cursor to the beginning of the 5th line':'X.seek(0); [X.readline() for i in range(4)]; X.readline()',
                        'move the file X cursor to the 5th byte of the 5th line':'X.seek(0); [X.readline() for i in range(4)]; X.readline().seek(5)',
                        'move the file X cursor to the end of the 5th line':'X.seek(0); [X.readline() for i in range(4)]; X.readline().seek(0,2)',
                        'move the file X cursor to the beginning of the 5th line':'X.seek(0); [X.readline() for i in range (4)]; X.readline().seek(0)',
                        'print the contents of the file X to the screen':'print(X.read())',
                        'print the first line of the file X to the screen':'print(X.readline())',
                        'open two files X and Y and create a new file Z with the contents of both':'with open(X) as f1, open(Y) as f2: with open(Z, "w") as f3: f3.write(f1.read() + f2.read())',
                        'open the file X and read the first 5 bytes':'with open(X) as f: f.read(5)'}
            action = "You got 10 right - touch your right ear."
            num2win = 10

        case 'pandas_operations':
            question = "Which command would you use to perform the following operation on the pandas dataframe X:\n XXX ?"
            optionsDict = {'create a new dataframe with the columns A and B from the dataframe X':'X[["A","B"]]',
                        'create a new dataframe with the rows 5 to 10 from the dataframe X':'X.iloc[5:11]',
                        'create a new dataframe with the columns A and B and the rows 5 to 10 from the dataframe X':'X[["A","B"]].iloc[5:11]',
                        'create a new dataframe with the columns A and B and the rows where the value in column A is greater than 5':'X[X["A"]>5][["A","B"]]',
                        'create a new dataframe with the columns A and B and the rows where the value in column A is greater than 5 and less than 10':'X[(X["A"]>5) & (X["A"]<10)][["A","B"]]',
                        'create a new dataframe with the columns A and B and the rows where the value in column A is greater than 5 or less than 10':'X[(X["A"]>5) | (X["A"]<10)][["A","B"]]',
                        'create a new dataframe with the columns A and B and the rows where the value in column A is not 5':'X[X["A"]!=5][["A","B"]]',
                        'create a new dataframe with the columns A and B and the rows where the value in column A is in the list Y':'X[X["A"].isin(Y)][["A","B"]]',
                        'create a new dataframe with the columns A and B and the rows where the value in column A is not in the list Y':'X[~X["A"].isin(Y)][["A","B"]]',
                        'create a new dataframe with the columns A and B and the rows where the value in column A is null':'X[X["A"].isnull()][["A","B"]]',
                        'create a new dataframe with the columns A and B and the rows where the value in column A is not null':'X[~X["A"].isnull()][["A","B"]]',
                        'show the first 5 rows of the dataframe X':'X.head()',
                        'write the dataframe X to a csv file called Y (e.g., "Y.csv")' : 'X.to_csv(Y)',
                        'print the column names of the dataframe X':'X.columns',
                        'get the number of rows in the dataframe X':'len(X)',
                        'get the number of columns in the dataframe X':'len(X.columns)',
                        'get the numbers of rows and columns in the dataframe X':'X.shape',
                        'find out what a function "function" in the dataframe X does':'help(X.function)',
                        'print out the last 5 rows of the dataframe X':'X.tail()',
                        'load a csv file called Y (e.g., "Y.csv") into a dataframe':'pd.read_csv(Y)',
                        'import the pandas library':'import pandas as pd',
                        'create a data frame from a numpy array Z with columns A and B':'pd.DataFrame(Z, columns=["A","B"])'}
            action = "You got 10 right - turn to face to the left."
            num2win = 10

        case 'for_and_while_loops':
            question = "Which command would you use to perform the following operations:\n XXX ?"
            optionsDict = {'print each element of the list X':'for i in X: print(i)',
                        'print each element of the list X with its index':'for i,j in enumerate(X): print(j,i)',
                        'print ten random integers between 1 and 100':'for i in range(10): print(random.randint(1,100))',
                        'print random integers between 1 and 100 until the number 5 is generated':'while True: i = random.randint(1,100); print(i); if i==5: break',
                        'exit the loop':'break',
                        'skip the rest of the code in the loop and move to the next iteration':'continue',
                        'skip to the next iteration if the iteration number is even':'if i%2==0: continue',
                        'exit the loop if the iteration number is even':'if i%2==0: break',
                        'continue without any effect if the number is even':'if i%2==0: pass',
                        'just keep going as if there was no command':'pass',
                        'print all values in the dictionary X':'for i in X.values(): print(i)',
                        'print all keys in the dictionary X':'for i in X.keys(): print(i)',
                        'print all key-value pairs in the dictionary X':'for i,j in X.items(): print(i,j)',
                        'count how many times the letter "a" appears in the string X':'sum(1 for i in X if i=="a")',
                        'count how many tries it takes to get a random number less than 5':'i = 0; while random.randint(1,100)>=5: i+=1; print(i)',
                        'count how many random numbers it takes to get a total greater than 100':'i = 0; total = 0; while t<=100: t += random.randint(1,100); i+=1; print(i)'}
            action = "You got 7 right - touch your right ear."
            num2win = 7

        case 'logic_statements':
            question = "Which command would you use to perform the following operations:\n XXX ?"
            optionsDict = {'check if the number X is greater than 5':'if X>5',
                        'check if the number X is less than 5':'if X<5',
                        'check if the number X is equal to 5':'if X==5',
                        'check if the number X is not equal to 5':'if X!=5',
                        'check if the number X is greater than or equal to 5':'if X>=5',
                        'check if the number X is less than or equal to 5':'if X<=5',
                        'check if the number X is between 5 and 10':'if 5<=X<=10',
                        'check if the number X is not between 5 and 10':'if not 5<=X<=10',
                        'check if the number X is between 5 and 10 or between 15 and 20':'if 5<=X<=10 or 15<=X<=20',
                        'check if the number X is between 5 and 10 and between 15 and 20':'if 5<=X<=10 and 15<=X<=20',
                        'check if the number X is not between 5 and 10 or between 15 and 20':'if not 5<=X<=10 or 15<=X<=20',
                        'check if the number X is not between 5 and 10 and between 15 and 20':'if not 5<=X<=10 and 15<=X<=20',
                        'check if the number X is positive':'if X>0',
                        'check if the number X is negative':'if X<0',
                        'check if the number X is even':'if X%2==0',
                        'check if the number X is odd':'if X%2!=0',
                        'check if the number X is a multiple of Y':'if X%Y==0',
                        'check if the number X is a power of Y':'if X == Y ** round(math.log(X, Y))',
                        'check if the number X is a prime number':'if all(X%i!=0 for i in range(2,X))',
                        'check if the string X is equal to the string Y':'if X==Y',
                        'check if two objects X and Y are the same object':'if X is Y',
                        'check if two lists X and Y have all same elements':'if X==Y',
                        'check if all elements of the list X are greater than 5':'if all(i>5 for i in X)',
                        'check if any element of the list X is greater than 5':'if any(i>5 for i in X)',
                        'check if all elements of the list X are strings':'if all(isinstance(i, str) for i in X)',
                        'check of all elements of the list X are in the list Y':'if all(i in Y for i in X)',
                        'check if the the string X is contained in the string Y':'if X in Y',
                        'check if the the string X is not contained in the string Y':'if X not in Y',
                        'print if a number is even, odd, or zero':'if X==0: print("zero"); elif X%2==0: print("even"); else: print("odd")',
                        'print if a number is positive, negative, or zero':'if X==0: print("zero"); elif X>0: print("positive"); else: print("negative")',
                        'print if a number is a multiple of 3, 5, both, or neither':'if X%3==0 and X%5==0: print("both"); elif X%3==0: print("3"); elif X%5==0: print("5"); else: print("neither")'}
            action = "You got 10 right - wave."
            num2win = 10

        case 'numpy_arrays':
            question = "Which command would you use to perform the following numpy operations:\n XXX ?"
            optionsDict = {'create a vector X with the values 1, 2, and 3':'X = np.array([1,2,3])',
                        'create a 3x3 matrix X with the values 1, 2, 3, 4, 5, 6, 7, 8, 9':'X = np.array([[1,2,3],[4,5,6],[7,8,9]])',
                        'create a matrix X with 3 rows and 4 columns of zeros':'X = np.zeros((3,4))',
                        'create a matrix X with 3 rows and 4 columns of ones':'X = np.ones((3,4))',
                        'create a matrix X with 3 rows and 4 columns of random numbers':'X = np.random.rand(3,4)',
                        'create a matrix X with 3 rows and 4 columns of random integers between 1 and 100':'X = np.random.randint(1,100,(3,4))',
                        'create a matrix X with 3 rows and 4 columns of normally distributed random numbers':'X = np.random.randn(3,4)',
                        'get the shape of the matrix X':'X.shape',
                        'get the number of rows in the matrix X':'X.shape[0]',
                        'get the number of columns in the matrix X':'X.shape[1]',
                        'get the number of dimensions in the matrix X':'X.ndim',
                        'get the number of elements in the matrix X':'X.size',
                        'get the data type of the elements in the matrix X':'X.dtype',
                        'get the sum of all elements in the matrix X':'X.sum()',
                        'get the sum of the rows in the matrix X':'X.sum(axis=1)',
                        'get the sum of the columns in the matrix X':'X.sum(axis=0)',
                        'get the mean of all elements in the matrix X':'X.mean()',
                        'get the mean of the rows in the matrix X':'X.mean(axis=1)',
                        'get the first 5 columns of the matrix X':'X[:,:5]',
                        'get the first 5 rows of the matrix X':'X[:5,:]',
                        'get the first 5 elements of the matrix X':'X[:5]',
                        'create a vector of 10 numbers from 1 to 10':'X = np.arange(1,11)',
                        'create a vector of uniformly spaced numbers from 1 to 5 with 10 elements':'X = np.linspace(1,5,10)',
                        'create a vector of 10 numbers from 1 to 10 in reverse order':'X = np.arange(10,0,-1)',
                        'create an identity matrix with 5 rows and 5 columns':'X = np.eye(5)',
                        'create a diagonal matrix with the values 1, 2, 3':'X = np.diag([1,2,3])',
                        'create a vector of log-spaced numbers from 1 to 5 with 10 elements':'X = np.logspace(0,np.log10(5),10)'}
            action = "You got 10 right - touch your left ear."
            num2win = 10

        case 'numpy_slicing':
            question = "Which command would you use to perform the following numpy operations:\n XXX ?"
            optionsDict = {'get the first element of the vector X':'X[0]',
                        'get the last element of the vector X':'X[-1]',
                        'get the first 3 elements of the vector X':'X[:3]',
                        'get the last 3 elements of the vector X':'X[-3:]',
                        'reshape the vector X into a 3x3 matrix':'X.reshape(3,3)',
                        'transpose the matrix X':'X.T',
                        'add a new row to the matrix X':'np.vstack((X,np.random.rand(1,X.shape[1])))',
                        'add a new column to the matrix X':'np.hstack((X,np.random.rand(X.shape[0],1)))',
                        'add a new dimension to the matrix X after the second existing dimension':'np.expand_dims(X,axis=2)',
                        'add a new dimension to the matrix X before the first existing dimension':'np.expand_dims(X,axis=0)',
                        'find the indices where the value in the matrix X is greater than 5':'np.where(X>5)',
                        'adds a new dimension to X so that it is at least 2D':'np.atleast_2d(X)',
                        'tile the matrix X 3 times along the first dimension':'np.tile(X,3,axis=0)',
                        'convert the matrix X to a vector':'X.flatten()',
                        'get the diagonal of the matrix X':'np.diag(X)',
                        'convert the matrix X to a list':'X.tolist()',
                        'convert the elements of the matrix X to integers':'X.astype(int)',
                        'convert the elements of the matrix X to strings':'X.astype(str)',
                        'convert the elements of the matrix X to floats':'X.astype(float)',
                        'get the second to the fourth element of the vector X':'X[1:4]',
                        'get the second to the last element of the vector X':'X[1:-1]',
                        'get the first element and the last element of the vector X':'np.array([X[0], X[-1]])',
                        'get the last element and the second to the last element of the vector X':'np.array([X[-1], X[-2]])',
                        'get the first 3 elements and the last 3 elements of the vector X':'np.concatenate((X[:3], X[-3:]))',
                        'get every third element starting with the first element of the vector X':'X[::3]',
                        'get every third element starting with the second element of the vector X':'X[1::3]',
                        'get a reversed vector with every third element starting with the last element of the vector X':'X[-1::-3]'}
            action = "You got 8 right - touch your nose with your right hand."
            num2win = 8

        case 'linear_equations':
            # The following questions will test students' knowledge of linear equations and how to solve them. Commands to consider are:
            # - np.linalg: solve, lstsq, matrix_rank, inv, cross, dot, det, inner, outer, matmul, trace, norm, cond, inv, pinv, svd 
            question = "Which command would you use to perform the following numpy operations:\n XXX ?"
            optionsDict = {'solve the linear equation AX = B for X':'np.linalg.solve(A,B)',
                        'find the least squares solution to the linear equation AX = B':'np.linalg.lstsq(A,B)',
                        'find the rank of the matrix A':'np.linalg.matrix_rank(A)',
                        'find the inverse of the matrix A':'np.linalg.inv(A)',
                        'find the cross product of the vectors A and B':'np.cross(A,B)',
                        'find the dot product of the vectors A and B':'np.dot(A,B)',
                        'find the determinant of the matrix A':'np.linalg.det(A)',
                        'find the inner product of the vectors A and B':'np.inner(A,B)',
                        'find the outer product of the vectors A and B':'np.outer(A,B)',
                        'find the matrix product of the matrices A and B':'np.matmul(A,B)',
                        'find the augmented matrix of the linear equation AX = B':'np.hstack((A,B))',
                        'find the rank of the augmented matrix of the linear equation AX = B':'np.linalg.matrix_rank(np.hstack((A,B)))',
                        'find the trace of the matrix A':'np.trace(A)',
                        'find the sum of the diagonal elements of the matrix A':'np.trace(A)',
                        'find the norm of the vector A':'np.linalg.norm(A)',
                        'find the condition number of the matrix A':'np.linalg.cond(A)',
                        'find the Moore-Penrose pseudo-inverse of the matrix A':'np.linalg.pinv(A)',
                        'find the singular value decomposition of the matrix A':'np.linalg.svd(A)'}
            action = "You got 10 right - touch your right ear."
            num2win = 10

        
        case 'class_objects':
            question = "Consider the class 'Pacemaker' to answer the following questions:\n XXX ?"
            optionsDict = {'Create a new pacemaker object with patient ID 123':'p = Pacemaker(123)',
                        'Create a new pacemaker object with patient ID 123 and pacing rate 80':'p = Pacemaker(123, 80)',
                        'Adjust the pacing rate of the pacemaker object p to 90':'p.adjust_rate(90)',
                        'Monitor the heart rate of the pacemaker object p with a current heart rate of 70':'p.monitor_heart(70)',
                        'What is the name of the constructor method in the Pacemaker class?':'__init__',
                        'What kind of method is "__init__" in the Pacemaker class?':'constructor',
                        'When called in a class, what does the "self" parameter refer to?':'the instance of the class',
                        'How does one access the pacing rate of the pacemaker object p?':'p.pacing_rate',
                        'How does one access the patient ID of the pacemaker object p?':'p.patient_id',
                        'How does the method "adjust_rate" access the pacing rate of the pacemaker object?':'self.pacing_rate',
                        'What is the default pacing rate of the pacemaker object p?':'70',
                        'Is a default value set for the patient ID in the Pacemaker class?':'No',
                        'What will happen if the patient ID is not provided when creating a new pacemaker object?':'An error will occur',
                        'What will happen if3 the "pacing_rate" is not set when creating a new pacemaker object?':'It will default to 70',
                        'What will happen if the pacing rate is set to 200 in the "adjust_rate" method?':'An error will occur'}
            action = "You got 7 right - Yay!!"
            num2win = 7

        case 'common_matrices':
            # This quiz looks at diff common matrices and their properties: zero, ones, symmetric, diagonal,
            # triangular, identity, orthogonal, positive definite.  Commands to consider are: np.zeros, np.ones,
            # np.eye, np.diag, np.triu, np.tril, np.identity, np.random.rand, np.random.randn, np.random.randint
            question = "Which command would you use to perform the following numpy operations:\n XXX ?"
            optionsDict = {'create a 3x3 matrix of zeros':'np.zeros((3,3))',
                        'create a 3x3 matrix of ones':'np.ones((3,3))',
                        'create a 3x3 identity matrix':'np.eye(3)',
                        'create a 3x3 matrix with the diagonal elements 1, 2, 3':'np.diag([1,2,3])',
                        'create a 3x3 upper triangular matrix from the matrix A':'np.triu(A)',
                        'create a 3x3 lower triangular matrix from the matrix A':'np.tril(A)',
                        'create a 3x3 symmetric matrix from the matrix A':'A + A.T',
                        'create a 3x3 orthogonal matrix':'Q = np.random.rand(3,3); Q, _ = np.linalg.qr(Q)',
                        'create a 3x3 positive definite matrix':'A = np.random.rand(3,3); A = np.dot(A,A.T)',
                        'create a 3x3 random matrix with values between 0 and 1':'np.random.rand(3,3)',
                        'check if a matrix is symmetric':'np.allclose(A, A.T)',
                        'check if a matrix is diagonal':'np.allclose(A, np.diag(np.diag(A)))',
                        'check if a matrix has full rank':'np.linalg.matrix_rank(A) == min(A.shape)',
                        'check if a matrix is positive definite':'np.all(np.linalg.eigvals(A) > 0)',
                        'check if a matrix is orthogonal':'np.allclose(np.dot(Q,Q.T), np.eye(3))'}
            action = "You got 10 right!"
            num2win = 10
        
        case 'matrix_norms':
            # This quiz tests students' knowledge of matrix properties such as range, null space, determinant, norms, and inverse.
            question = "Answer the fowllowing questions about matrix properties and numpy operations:\n XXX ?"
            optionsDict = {'What command would you use to find the rank of the matrix A':'np.linalg.matrix_rank(A)',
                        'What relationship tells us that v is in the null space of A':'A @ v == 0',
                        'If A is a (mxn) matrix with rank r, what is the dimension of the null space of A':'n - r',
                        'If A is a (mxn) matrix with rank r, what is the dimension of the range of A':'r',
                        'What command would you use to find the determinant of the matrix A':'np.linalg.det(A)',
                        'If the null space of A is non-trivial, what else can we say about the determinant of A':'The determinant of A is zero',
                        'If the determinant of A is zero, what else can we say about the matrix A':'The matrix A is singular',
                        'What command would you use to find the Frobenius norm of the matrix A':'np.linalg.norm(A)',
                        'What command would you use to find the inverse of the matrix A':'np.linalg.inv(A)',
                        'If the matrix A has inverse A^-1, what is the relationship between A and its inverse':'A @ A^1 = I',
                        'If the matrix A is orthonormal, what is the relationship between A and its transpose':'A @ A.T = I',
                        'If the matrix A is symmetric, what is the relationship between A and its transpose':'A = A.T',
                        'If the matrix A is diagonal, what is the relationship between A and its transpose':'A = A.T',
                        'If the matrix A is positive definite, what is the relationship between A and its transpose':'A = A.T',
                        'If the square matrix A is singular, what can we say about its determinant':'The determinant of A is zero',
                        'If the square matrix is invertible, what can we say about its determinant':'The determinant of A is non-zero',
                        'If the (mxn) matrix A has full column rank, what is the relationship between m and n':'m >= n',
                        'If the spectral norm of A is less than 1, what can we say about v and Av':'The vector v is shortened by the transformation A',
                        'If the spectral norm of A is greater than 1, what can we say about v and Av':'The vector v is lengthened by the transformation A',
                        'If the matrix A is orthonormal, what can be say about about v and Av':'The transformation A preserves the length of the vector v',
                        'If v = [1,2,-3], what is the 2-norm of v':'|v| = sqrt(14)',
                        'If v = [1,2,-3], what is the 1-norm of v':'|v| = 6',
                        'If v = [1,2,-3], what is the infinity-norm of v':'|v| = 3',
                        'Which norm is the absolute value of the largest element in the vector':'Infinity norm',
                        'Which norm is the square root of the sum of the squares of the elements in the vector':'2-norm',
                        'Which norm is the sum of the absolute values of the elements in the vector':'1-norm',
                        'Which norm is also known as the Euclidean norm':'2-norm',
                        'What is the definiton of an induced matrix norm':'max(|Ax|/|x|) for all x != 0',
                        'How do you find the spectral norm of a matrix':'max(svd(A))',
                        'Which induced norm is the same as the spectral norm':'2-norm',
                        'What is the formular for the Moore-Penrose pseudo-inverse of a matrix A':'(A.T @ A)^-1 @ A.T',
                        'When does the inverse of a square matrix A exist':'When the determinant of A is non-zero',
                        'When does the inverse of a square matrix A not exist':'When the determinant of A is zero'}
            action = "You got 10 right!"
            num2win = 10
        
        case 'eigenvalues_and_eigenvectors':
            # This quiz tests students' knowledge of eigenvalues and eigenvectors of matrices. Commands to consider are:
            # - np.linalg: eig, eigvals, eigvalsh, eigvals, eigvalsh, eigvecs, eigvecsh
            question = "Answer the following questions about eigenvalues and eigenvectors of matrices:\n XXX ?"
            optionsDict = {'What command would you use to find the eigenvalues and eigenvectors of the matrix A':'np.linalg.eig(A)',
                        'What is the relationship between the matrix A and its eigenvectors and eigenvalues':'A @ v = λv',
                        'What do you know about the eigenvectors of a symmetric matrix':'The eigenvectors are orthogonal',
                        'What do you know about the eigenvalues of a symmetric matrix':'The eigenvalues are real',
                        'What do you know about the eigenvalues of a positive definite matrix':'The eigenvalues are positive',
                        'If the square matrix A has a non-trivial null space, what can we say about its eigenvalues':'The matrix A has at least one eigenvalue of zero',
                        'If the determinant of the square matrix A is zero, what can we say about its eigenvalues':'The matrix A has at least one eigenvalue of zero',
                        'If the square matrix A is singular, what can we say about its eigenvalues':'The matrix A has at least one eigenvalue of zero',
                        'If the square matrix A is invertible, what can we say about its eigenvalues':'The matrix A has no eigenvalues of zero',
                        'What are the eigenvalues of the identity matrix':'The eigenvalues are all 1',
                        'What are the eigenvectors of the identity matrix':'Any non-zero vector',
                        'What are the eigenvalues of a diagonal matrix':'The diagonal elements',
                        'What are the eigenvectors of a diagonal matrix':'The standard basis vectors',
                        'What are the eigenvalues of an upper triangular matrix':'The diagonal elements',
                        'What would be the overal transformation combining transformation matrices M1, M2, and M3 in that order':'M3 @ M2 @ M1',
                        'If a transformation matrix A results in a pure rotation, what can we say about its eigenvalues':'All eigenvalues have unit magnitude',
                        'What are the eigenvectors of an upper triangular matrix':'The standard basis vectors'}
            action = "You got 10 right!"
            num2win = 10
        
        case "linear_transformations_and_svd":
            question = "Answer the following questions about linear transformations and singular value decomposition (SVD):\n XXX ?"
            optionsDict = {'What command would you use to find the singular value decomposition of the matrix A':'np.linalg.svd(A)',
                        'What is the relationship between the matrix A and its singular value decomposition':'A = U @ S @ V.T',
                        'What do the singular values represent in the SVD of a matrix':'The scaling factors of the transformation',
                        'What do the columns of the matrix U represent in the SVD of a matrix':'The left singular vectors',
                        'What do the columns of the matrix V represent in the SVD of a matrix':'The right singular vectors',
                        'What is the basis for the range of a matrix A in the SVD':'The columns of U corresponding to non-zero singular values',
                        'What is the basis for the null space of a matrix A in the SVD':'The columns of V corresponding to zero singular values',
                        'What is the relationship between the singular values and the eigenvalues of a matrix':'The singular values are the square roots of the eigenvalues of A.T @ A',
                        'If a (nxn) matrix has a rank of r, how many non-zero singular values will it have':'r',
                        'If a (nxn) matrix has a rank of r, how many zero singular values will it have':'n - r',
                        'What is the relationship between the singular values and the eigenvalues of a symmetric matrix':'The singular values are the absolute values of the eigenvalues',
                        'If A = U @ S @ V.T, what is A.T':'V @ S.T @ U.T',
                        'If A = U @ S @ V.T, what is A^-1':'V @ S^-1 @ U.T',
                        'If A = U @ S @ V.T, what is the Moore-Penrose pseudo-inverse of A':'V @ S^-1 @ U.T'}
            action = "You got 10 right!"
            num2win = 10

        case _:
            print("Quiz not found")
            return
    multiple_choice(optionsDict, question, action, num2win)


### Pacemaker Class in Python
class Pacemaker:
    """
    A simple representation of a pacemaker used in biomedical engineering.
    """

    def __init__(self, patient_id, pacing_rate=70):
        """
        Initialize the pacemaker with a patient ID and a default pacing rate (bpm).
        """
        self.patient_id = patient_id
        self.pacing_rate = pacing_rate  # Default 70 beats per minute

    def adjust_rate(self, new_rate):
        """
        Adjust the pacing rate of the pacemaker.
        """
        if 40 <= new_rate <= 180:
            self.pacing_rate = new_rate
            print(f"Pacemaker rate set to {self.pacing_rate} bpm for Patient {self.patient_id}.")
        else:
            print("Error: Rate must be between 40 and 180 bpm.")

    def monitor_heart(self, current_heart_rate):
        """
        Simulate heart monitoring. If the heart rate is too low, the pacemaker activates.
        """
        if current_heart_rate < self.pacing_rate:
            print(f"Pacemaker activated for Patient {self.patient_id} to maintain {self.pacing_rate} bpm.")
        else:
            print(f"Heart rate normal ({current_heart_rate} bpm). No pacemaker intervention needed.")