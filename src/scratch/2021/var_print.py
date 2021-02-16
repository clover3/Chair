from log_lib import log_variables

g_variable = 1


def main():
    my_variable = 10
    # print(varname(g_variable))
    # print(varname(my_variable))
    #
    avkai = 0
    log_variables(g_variable)


    log_variables(g_variable, my_variable)


if __name__ == "__main__":
    main()