
//----Distance matrix----//
Distance in decimeter between all locations in the warehouse, with the depot occupying the first location. E.g., the first line contains the distance from the depot to all locations, where the first number is the distance from the depot to the depot (=0), the second number the distance from the depot to the first location, etc. The second line contains the distance from the first location to all locations, starting with the depot.

//----Static problem----//
A file with a summary of the solution of every instance, solved by both the iterated local search and large neighbourhood search, is found in 'Results LNS and ILS.xlsx'

###Instances###
Contains the instances of the static problem. These instances are included for completeness, but they are NOT made by us. Instead, they are copied from the paper of Van Gils, T., Caris, A., Ramaekers, K. & Braekers, K. Formulating and Solving the Integrated Batching, Routing, and Picker Scheduling Problem in a Real-life Spare Parts Warehouse (2019). European Journal of Operational Research, 277 (3), 814-830. 10.1016/j.ejor.2019.03.012 . 
These files can also be downloaded from their original location: https://www.uhasselt.be/Datasets-and-results

Every instance consists of two files: an order list and an order line list. The file name is constructed as follows: OrderList_LargeProblems_InstanceNo_Replication. Every factor combination from the factorial design has a different instance number, while for a specific factor combination 30 different instances (=replications) are made.

OrderList files contain the following columns: OrderID NumberOfOrderLines DueTime FirstOrderLineID

OrderLineList files contain the following columns: OrderID OrderLineID AisleID CellID LevelID LocationID

###Results ILS###
Contains the detailed results of every instance after running the iterated local search algorithm. Shows how all orders were batched, which batches were assigned to every picker and in which sequence the batches were picked. 

###Results LNS###
Contains the detailed results of every instance after running the large neighbourhood search algorithm. Shows how all orders were batched, which batches were assigned to every picker and in which sequence the batches were picked. 


//----Online problem----//
A file with a summary of the solution of every instance, solved by the large neighbourhood search algorithm is found in 'Results online problem instances.xlsx'

###Instances###
Every instance consists of two files: an order list and an order line list. The file name is constructed as follows: OrderList_LargeProblems_InstanceNumber_ReplicationNumber. Every factor combination from the factorial design has a different instance number, while for a specific factor combination 10 different instances (=replications) are made (starting at 21, going to 30).

OrderList files contain the following columns: OrderID NumberOfOrderLines ReleaseTime DueTime FirstOrderLineID

OrderLineList files contain the following columns: OrderID OrderLineID AisleID CellID LevelID LocationID

###Parameter tuning###
'Resuls parameter tuning.xlsx' contains the summary of the solution of all instances shown in the parameter tuning 'Instances' folder, for the different settings tested during the parameter tuning.

###Detailed results"###
Contains the solution file of every instance. The name is interpreted as follows: PerformedSolutionHeuristic_LargeProblems__InstanceNumber_ReplicationNumber_PercentageOfTrucksActive

These solution files contain the final solution, as performed by the order pickers.