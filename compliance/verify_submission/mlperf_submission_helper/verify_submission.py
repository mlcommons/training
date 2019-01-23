import argparse
import os
import shutil
import sys

import checks as submission_checks
import constants
import report


def verify_submission(args):
  root_dir = args.root
  public_key = args.public_key
  private_key = args.private_key
  encrypt_out = args.encrypt_out
  decrypt_out = args.decrypt_out

  # validate args
  if any([public_key, encrypt_out]) and not all([public_key, encrypt_out]):
    print("--encrypt-key and --encrypt-out must be present togetger.")
    sys.exit(1)
  if any([private_key, decrypt_out]) and not all([private_key, decrypt_out]):
    print("--decrypt-key and --decrypt-out must be present together.")
    sys.exit(1)
  if all([private_key, public_key]):
    print("--encrypt-key and --decrypt-key cannot be present together.")
    sys.exit(1)

  if any([public_key, private_key]):
    import crypto

  # if decrypt-key is provided, then decrypt the submission, save it to
  # decrypt-out and point submission root to the decrypted directory
  if private_key:
    try:
      crypto.decrypt_submission(private_key, root_dir, decrypt_out)
    except Exception as e:
      print("Unable to decrypt submission: {}".format(str(e)))
      sys.exit(1)
    print("Decrypted submission saved at {}".format(decrypt_out))
    root_dir = decrypt_out

  # perform verifications and extract results
  checks = submission_checks.SubmissionChecks()
  checks.verify_dirs_and_files(root_dir)
  checks.verify_metadata()
  checks.compile_results()

  checks.report.print_report()
  checks.report.print_results()

  # if encrypt-key is provided, then encrypt the submission
  # and save it to encrypt-out
  if public_key:
    try:
      crypto.encrypt_submission(public_key, root_dir, encrypt_out)
    except Exception as e:
      print("Unable to encrypt submission: {}".format(str(e)))
      sys.exit(1)
    print("Encrypted submission saved at {}".format(encrypt_out))


def main():
  parser = argparse.ArgumentParser(description="Verify MLPerf submission.")
  parser.add_argument(
      "root", metavar="SUBMISSION_ROOT", help="submission root directory")
  parser.add_argument(
      "--encrypt-key",
      dest="public_key",
      default=None,
      help="public key for encrypting log files")
  parser.add_argument(
      "--encrypt-out",
      dest="encrypt_out",
      default=None,
      help="output path for encrypted submission")
  parser.add_argument(
      "--decrypt-key",
      dest="private_key",
      default=None,
      help="private key for decrypting log files")
  parser.add_argument(
      "--decrypt-out",
      dest="decrypt_out",
      default=None,
      help="output path for decrypted submission")
  args = parser.parse_args()

  verify_submission(args)


if __name__ == "__main__":
  main()
