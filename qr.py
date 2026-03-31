import qrcode

url = input("Enter URL: ").strip()
filename = input("Enter output file name (example: my_qr.png): ").strip()

if not filename:
    filename = "qr_code.png"

img = qrcode.make(url)
img.save(filename)

print(f"QR code saved as {filename}")