import fnmatch
import os
import shutil

from Cryptodome.PublicKey import RSA
from Cryptodome.Random import get_random_bytes
from Cryptodome.Cipher import AES, PKCS1_OAEP


def encrypt_file(public_key, src_file, dest_file):
  try:
    with open(src_file) as f:
      rsa_key = RSA.import_key(open(public_key).read())
      session_key = get_random_bytes(16)
      # Encrypt session key
      cipher_rsa = PKCS1_OAEP.new(rsa_key)
      encrypted_session_key = cipher_rsa.encrypt(session_key)
      # Encrypt data
      cipher_aes = AES.new(session_key, AES.MODE_EAX)
      ciphertext, tag = cipher_aes.encrypt_and_digest(f.read().encode("utf-8"))
  except Exception as e:
    print("Unable to encrypt file: {}".format(src_file))
    raise e

  try:
    with open(dest_file, "wb") as f:
      for x in (encrypted_session_key, cipher_aes.nonce, tag, ciphertext):
        f.write(x)
  except Exception as e:
    print("Unable to write output file {}".format(dest_file))
    raise e


def decrypt_file(private_key, src_file, dest_file):
  try:
    with open(src_file, "rb") as f:
      rsa_key = RSA.import_key(open(private_key).read())
      encrypted_session_key = f.read(rsa_key.size_in_bytes())
      nonce = f.read(16)
      tag = f.read(16)
      ciphertext = f.read(-1)

      # Decrypt session key
      cipher_rsa = PKCS1_OAEP.new(rsa_key)
      session_key = cipher_rsa.decrypt(encrypted_session_key)
      # Decrypt data
      cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
      data = cipher_aes.decrypt_and_verify(ciphertext, tag)
      data = data.decode("utf-8")
  except Exception as e:
    print("Unable to decrypt file: {}".format(src_file))
    raise e

  try:
    with open(dest_file, "w") as f:
      f.write(data)
  except Exception as e:
    print("Unable to write output file: {}".format(dest_file))
    raise e


def encrypt_submission(key, src_dir, dest_dir):
  if os.path.isdir(dest_dir):
    raise Exception("Output directory already exists.")
  os.mkdir(dest_dir, mode=0o755)
  for root, dirs, files in os.walk(src_dir):
    # identify result files and encrypt, else directly copy
    if fnmatch.fnmatch(root, os.path.join(src_dir, "results", "*", "*")):
      for f in files:
        from_file = os.path.join(root, f)
        to_file = from_file.replace(
            src_dir.rstrip(os.sep), dest_dir.rstrip(os.sep), 1)
        encrypt_file(key, from_file, to_file)
    else:
      for d in dirs:
        from_dir = os.path.join(root, d)
        to_dir = from_dir.replace(
            src_dir.rstrip(os.sep), dest_dir.rstrip(os.sep), 1)
        os.mkdir(to_dir, mode=0o755)
      for f in files:
        from_file = os.path.join(root, f)
        to_file = from_file.replace(
            src_dir.rstrip(os.sep), dest_dir.rstrip(os.sep), 1)
        shutil.copyfile(from_file, to_file)


def decrypt_submission(key, src_dir, dest_dir):
  if os.path.isdir(dest_dir):
    raise Exception("Output directory already exists.")
  os.mkdir(dest_dir, mode=0o755)
  for root, dirs, files in os.walk(src_dir):
    # identify result files and encrypt, else directly copy
    if fnmatch.fnmatch(root, os.path.join(src_dir, "results", "*", "*")):
      for f in files:
        from_file = os.path.join(root, f)
        to_file = from_file.replace(
            src_dir.rstrip(os.sep), dest_dir.rstrip(os.sep), 1)
        decrypt_file(key, from_file, to_file)
    else:
      for d in dirs:
        from_dir = os.path.join(root, d)
        to_dir = from_dir.replace(
            src_dir.rstrip(os.sep), dest_dir.rstrip(os.sep), 1)
        os.mkdir(to_dir, mode=0o755)
      for f in files:
        from_file = os.path.join(root, f)
        to_file = from_file.replace(
            src_dir.rstrip(os.sep), dest_dir.rstrip(os.sep), 1)
        shutil.copyfile(from_file, to_file)
