import os

import crypto


def main():
    pub_key = "../examples/id_rsa_1024.pub"
    priv_key = "../examples/id_rsa_1024"
    raw_file = "../examples/example_log.txt"
    encrypted_file = "../examples/example_encrypted_file.txt"
    decrypted_file = "../examples/example_decrypted_file.txt"
    raw_subm = "../../example_submission"
    encrypted_subm = "../examples/encrypted_subm"
    decrypted_subm = "../examples/decrypted_subm"

    # crypto.encrypt_file(pub_key, raw_file, encrypted_file)
    # crypto.decrypt_file(priv_key, encrypted_file, decrypted_file)
    crypto.encrypt_submission(pub_key, raw_subm, encrypted_subm)
    crypto.decrypt_submission(priv_key, encrypted_subm, decrypted_subm)


if __name__ == "__main__":
    main()
