# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IdXym0QuOgiE5LK34sDJEnZxzaBfqRL0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IdXym0QuOgiE5LK34sDJEnZxzaBfqRL0" -O models.zip && rm -rf /tmp/cookies.txt
gdown --id 1fLYHoGIQJoubNq4dA8ZmZ7pMOc8ax-GX --output models.zip
unzip models.zip


