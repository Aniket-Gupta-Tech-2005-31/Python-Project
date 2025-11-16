def water_jug():
    jug1, jug2 = 0, 0
    cap1, cap2 = 3, 4
    goal = 2

    print("Jug1\tJug2")
    while True:
        print(jug1, "\t", jug2)
        if jug1 == goal or jug2 == goal:
            print("Goal reached!")
            break

        elif jug2 == 0:
            jug2 = cap2

        elif jug1 < cap1:
            transfer = min(jug2, cap1 - jug1)
            jug1 += transfer
            jug2 -= transfer
            
        elif jug1 == cap1:
            jug1 = 0

water_jug()
