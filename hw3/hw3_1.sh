# TODO: create shell script for running your ViT testing code
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1u_4bMyHOFr_s2-a8JrFERKIerbirMEcr' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1u_4bMyHOFr_s2-a8JrFERKIerbirMEcr" -O best_vit.pth
# Example
python3 hw3_1.py --folder $1 --save_csv $2
