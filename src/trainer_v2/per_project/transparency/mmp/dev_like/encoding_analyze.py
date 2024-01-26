l = range(768, 879)

l = range(0x0300, 0x0400)


print("youÃ¢re", "youÃ¢re".encode("utf-8"))
target = "An Interactive History-is"
encoding = "Windows-1252"
bytes1 = target.encode(encoding)
print(bytes1)
listTestByteAsHex = [hex(x) for x in bytes1]
print(bytes1, listTestByteAsHex)
s2 = bytes1.decode("utf-8")

print()
# byte_seq = b'\x97\x69\x73'
byte_seq = b'\xb4\x74\x20'
print(byte_seq)
print(byte_seq.decode(encoding))
print(byte_seq.decode('utf-8', errors='ignore'))
print(target, bytes1, s2)
#
# for b in byte_seq:
#     try:
#         other_encode = b.dec(encoding)
#     except UnicodeEncodeError:
#         other_encode = "Undef"
#
#     arr = [
#         i,
#         int.to_bytes(i, 2, "big"),
#         chr(i),
#         chr(i).encode("utf-8"),
#         other_encode,
#     ]
#     print("\t".join(map(str, arr)))

# int.from_bytes(chr(i).encode("utf-8"), "little")