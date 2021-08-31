from AllocationFuncs.graph_building_funcs import fat_tree

fat_tree(save='./graph.txt',
            server_per_rack=16,
            number_of_racks=32,
            rack_to_aggregate_link=4,
            aggregate_to_rack_link=8,
            aggregate_to_core_link=1,
            core_to_aggregate_link=2,
            server_label='Resource',
            rack_label='Rack',
            aggregate_label='Aggregate',
            core_label='Core',
            link_label='Link',
            show=True)