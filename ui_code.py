from tkinter import *
from tkinter import ttk
from code import * #code file
labels = []
answer = []
root = Tk()
root.geometry('800x700')
root.configure(background='red')
label = Label(root, text="Vector Space Model",bg="green",font = "Helvetica",fg="pink")
label.pack()
label.config(font=("Courier", 20))

label.pack()

inputtxt = Text(root, height=5, width=52)
inputtxt.pack()

b1 = Button(root, text="Submit Query", command=lambda: Take_input(),bg="grey")
b1.pack()
b1.config(font=("Courier", 12))

scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

mylist = Listbox(root, yscrollcommand=scrollbar.set,bg="blue",fg="yellow",activestyle = 'dotbox', 
                  font = "Helvetica",height=5,width=50,justify=CENTER,bd=5)


mylist.pack(side=RIGHT, fill=BOTH)
scrollbar.config(command=mylist.yview)
def remove_label():
    mylist.delete(0, 'end')
    


def Take_input():
    global answer
    global doc_list
    global cosine_sim_list
    INPUT = inputtxt.get("1.0", "end-1c")
    print(INPUT)
    cosine_sim_list,doc_list = takeInput(INPUT)
    print(cosine_sim_list)
    print(doc_list)
    remove_label()
    for i in range(len(doc_list)):
        mylist.insert(END, "Doc ID: " + str(doc_list[i]), "Cosin Sim: " + str(cosine_sim_list[i]))
    # for i in range(len(cosine_sim_list)):
    #     mylist.insert(END, "Cosin Sim: " + str(cosine_sim_list[i]))


root.mainloop()
