import pyotp

key = '2UD7MV4DFPUPO26PVPCKIEP3CDZNYR5M'
totp = pyotp.TOTP(key)
print(totp.now())