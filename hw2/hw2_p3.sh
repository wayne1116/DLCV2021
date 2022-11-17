# TODO: create shell script for running your DANN model
USPS="usps"
MNISTM="mnistm"
SVHN="svhn"

if [[ "$2" == "$USPS" ]]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14atkt-k2sIUkaMxmaCX5dW70yjcNgGe_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14atkt-k2sIUkaMxmaCX5dW70yjcNgGe_" -O extractor_model_m2u.pth
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eeZ4-wya8eUYBojd6co3AWR9kMB693mv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eeZ4-wya8eUYBojd6co3AWR9kMB693mv" -O predictor_model_m2u.pth
elif [[ "$2" == "$MNISTM" ]]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bl3MtGe7oBHeoQESR3leulgbrU7L6neO' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bl3MtGe7oBHeoQESR3leulgbrU7L6neO" -O extractor_model_s2m.pth
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16fN77jXfarb-rVcMSYbzzdllD7sop-8d' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16fN77jXfarb-rVcMSYbzzdllD7sop-8d" -O predictor_model_s2m.pth
else
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jKTm2muGjNnwBdYz-eVSgHOC3mypFzm7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jKTm2muGjNnwBdYz-eVSgHOC3mypFzm7" -O extractor_model_u2s.pth
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Vmad0aPplSnqmAXWRMfn5v_F3zzTQSc2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Vmad0aPplSnqmAXWRMfn5v_F3zzTQSc2" -O predictor_model_u2s.pth
fi
# Example
python3 p3.py --test_path $1 --target_domain $2 --output_csv $3