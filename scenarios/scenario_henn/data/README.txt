This folder is divided into four groups abc1, abc2, ran1, and ran2 corresponding to the four warehouses used in these experiments.

Each warehouse contains two types of instances: those that describe the layout of the warehouse and those that describe the orders with their items:

- For the first ones:
    * They are named as: sett<number_of_setting>.txt where:
        + <number_of_setting> is an integer number related to an specific layout configuration of the warehouse
    * The structure of the setting file is described in description.pdf file

- For the second ones:
    * They are named as: <setting_number>[s|l]-<number_of_orders>-<picker_capacity>-<number_of_instance>.txt where:
        + <setting_number> is the number of the setting
        + [s|l] is inherited from the set of instances of Sebastian Henn
        + <number_of_orders> is the number of orders in that instance (40, 60, 80, 100)
        + <picker_capacity> is the picker capacity (30, 45, 60, 75)
        + <number_of_instance> is the number of the instance (just 0 in this case)
    * The important lines are:
        + For every order we have: Order <number_of_order> number of articles <number_of_articles> where:
            -- <number_of_order> is the number of the order
            -- <number_of_articles> is the number of articles in that order
        + And a list of <number_of_articles> with the following information:
            -- <number_of_item_inside_the_order> is an integer number with the number of the item
            -- Aisle <number_of_aisle>, being <number_of_aisle> the number of the aisle in which the article is located (taking into account that there are twice aisles than in reality, that is, if a warehouse has 4 aisles, the first aisle would be 0 and the last aisle, 7)
            -- Location <position_in_aisle>, being <position_in_aisle> the storage location of the item within an aisle, from 0 to length of aisle
