# mlperf-submission-helper

## Key generation

If you need to encrypt your submission, run the followings to generate a key pair.

```
ssh-keygen -t rsa -b 1024 -f ${KEY_NAME} -N ''
```
where `${KEY_NAME}` is the name of your key files. You can give any valid file name.

In the generated files, the private key should be named `${KEY_NAME}`, the public key is named `${KEY_NAME}.pub`.

## Install

```
git clone --recurse-submodules https://github.com/mlperf/training.git
cd training/compliance/verify_submission

# If you need encryption/decryption:
pip install -r crypto_requirements.txt
```

## Run

- Basic verification (no encryption/decryption):

```
python mlperf_submission_helper/verify_submission.py ${SUBMISSION_ROOT}
```
where `${SUBMISSION_ROOT}` is the root directory of your submission.

- Verification with encryption (the submission will be verified and then encrypted and saved at a new directory):

```
python mlperf_submission_helper/verify_submission.py \
--encrypt-key ${PUBLIC_KEY} --encrypt-out ${ENCRYPT_OUT} ${SUBMISSION_ROOT}
```
where `${PUBLIC_KEY}` is the path to your public key, `${ENCRYPT_OUT}` is the path to encrypted submission (it should be a new directory).

- Verification with decryption (the submission will be decrypted and saved at a new directory, and then the decrypted submission will be verified):

```
python mlperf_submission_helper/verify_submission.py \
--decrypt-key ${PRIVATE_KEY} --decrypt-out ${DECRYPT_OUT} ${SUBMISSION_ROOT}
```
where `${PRIVATE_KEY}` is the path to your private key, `${DECRYPT_OUT}` is the path to decrypted submission (it should be a new directory).
