
def write_intervals(chromosome, length):
    start = 0
    increment = 57344
    end = 114688
    with open(f"metadata/chr{chromosome}_intervals.txt", 'a') as f:
        while end <= length:
            f.write(f'"chr{chromosome}_{start}_{end}"\n')
            start += increment
            end += increment
        f.write(f'"chr{chromosome}_{start}_{end}"')

write_intervals(chromosome = 8, length = 145138636)
write_intervals(chromosome = 9, length = 138394717)
write_intervals(chromosome = 10, length = 133797422)
write_intervals(chromosome = 11, length = 135086622)
