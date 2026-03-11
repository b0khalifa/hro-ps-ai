import pandas as pd

def schedule_operations(num_surgeries, rooms_available):

    surgeries_per_room = num_surgeries // rooms_available
    extra = num_surgeries % rooms_available

    schedule = []

    for i in range(rooms_available):
        surgeries = surgeries_per_room
        if i < extra:
            surgeries += 1

        schedule.append({
            "room": i+1,
            "surgeries": surgeries
        })

    return pd.DataFrame(schedule)