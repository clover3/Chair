import tkinter as tk


def run_gui():
    def process_text(seq2seq):
        input_text = input_field.get("1.0", "end-1c")
        output_text = seq2seq(input_text)
        output_field.delete("1.0", "end")
        output_field.insert("1.0", output_text)

    root = tk.Tk()

    input_label = tk.Label(root, text="Input:")
    input_label.grid(row=0, column=0)

    input_field = tk.Text(root, height=10, width=50)
    input_field.grid(row=0, column=1)

    output_label = tk.Label(root, text="Output:")
    output_label.grid(row=1, column=0)

    output_field = tk.Text(root, height=10, width=50)
    output_field.grid(row=1, column=1)

    send_button = tk.Button(root, text="Send", command=process_text)
    send_button.grid(row=2, column=1)

    root.mainloop()


def main():
    run_gui()


if __name__ == "__main__":
    main()