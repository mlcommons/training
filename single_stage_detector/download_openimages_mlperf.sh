#!/bin/bash

: "${DATASET_PATH:=/datasets/open-images-v6}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --dataset-path  )        shift
                                      DATASET_PATH=$1
                                      ;;
    esac
    shift
done

MLPERF_CLASSES=('Accordion' 'Airplane' 'Alpaca' 'Ant' 'Antelope' 'Apple' 'Backpack' 'Bagel' 'Balloon'
'Banana' 'Barge' 'Barrel' 'Baseball bat' 'Baseball glove' 'Bat (Animal)' 'Bathtub' 'Bee' 'Beehive'
'Beer' 'Bell pepper' 'Bench' 'Bicycle' 'Bicycle helmet' 'Bicycle wheel' 'Billboard'
'Billiard table' 'Book' 'Bookcase' 'Boot' 'Bottle' 'Bow and arrow' 'Bowl' 'Box' 'Boy'
'Brassiere' 'Bread' 'Broccoli' 'Bronze sculpture' 'Brown bear' 'Bull' 'Bus' 'Bust'
'Butterfly' 'Cabinetry' 'Cake' 'Camel' 'Camera' 'Candle' 'Candy' 'Cannon' 'Canoe' 'Carrot'
'Cart' 'Castle' 'Cat' 'Caterpillar' 'Cattle' 'Cello' 'Chair' 'Cheetah' 'Chest of drawers'
'Chicken' 'Chopsticks' 'Christmas tree' 'Coat' 'Cocktail' 'Coconut' 'Coffee' 'Coffee cup'
'Coffee table' 'Coin' 'Computer keyboard' 'Computer monitor' 'Computer mouse'
'Convenience store' 'Cookie' 'Countertop' 'Cowboy hat' 'Crab' 'Crocodile' 'Crown' 'Cucumber'
'Cupboard' 'Curtain' 'Deer' 'Desk' 'Dice' 'Dinosaur' 'Dog' 'Doll' 'Dolphin' 'Door'
'Door handle' 'Doughnut' 'Dragonfly' 'Drawer' 'Dress' 'Drum' 'Duck' 'Eagle' 'Earrings'
'Egg (Food)' 'Elephant' 'Falcon' 'Fedora' 'Fireplace' 'Flag' 'Flowerpot' 'Football'
'Football helmet' 'Fork' 'Fountain' 'Fox' 'French fries' 'Frog' 'Gas stove' 'Giraffe' 'Girl'
'Glasses' 'Goat' 'Goggles' 'Goldfish' 'Golf cart' 'Gondola' 'Goose' 'Grape' 'Grapefruit'
'Guitar' 'Hamburger' 'Hamster' 'Handbag' 'Handgun' 'Harbor seal' 'Headphones' 'Helicopter'
'High heels' 'French horn' 'Horse' 'House' 'Houseplant' 'Human arm' 'Human beard' 'Human ear'
'Human eye' 'Human face' 'Human foot' 'Human hair' 'Human hand' 'Human head' 'Human leg'
'Human mouth' 'Human nose' 'Ice cream' 'Jacket' 'Jaguar (Animal)' 'Jeans' 'Jellyfish' 'Jet ski'
'Jug' 'Juice' 'Kangaroo' 'Kettle' 'Kitchen & dining room table' 'Kite' 'Knife' 'Ladder'
'Ladybug' 'Lamp' 'Lantern' 'Laptop' 'Lavender (Plant)' 'Lemon' 'Leopard' 'Light bulb'
'Lighthouse' 'Lily' 'Lion' 'Lizard' 'Lobster' 'Loveseat' 'Man' 'Maple' 'Mechanical fan'
'Microphone' 'Miniskirt' 'Mirror' 'Missile' 'Mobile phone' 'Monkey' 'Motorcycle' 'Mouse'
'Muffin' 'Mug' 'Mule' 'Mushroom' 'Musical keyboard' 'Necklace' 'Nightstand' 'Office building'
'Orange' 'Ostrich' 'Otter' 'Oven' 'Owl' 'Oyster' 'Paddle' 'Palm tree' 'Pancake' 'Parachute'
'Parrot' 'Pasta' 'Peach' 'Pear' 'Pen' 'Penguin' 'Piano' 'Picture frame' 'Pig' 'Pillow'
'Pineapple' 'Pizza' 'Plastic bag' 'Plate' 'Platter' 'Polar bear' 'Pomegranate' 'Porch'
'Poster' 'Potato' 'Pumpkin' 'Rabbit' 'Radish' 'Raven' 'Refrigerator' 'Rhinoceros' 'Rifle'
'Rocket' 'Roller skates' 'Rose' 'Salad' 'Sandal' 'Saucer' 'Saxophone' 'Scarf' 'Scoreboard'
'Sea lion' 'Sea turtle' 'Segway' 'Shark' 'Sheep' 'Shelf' 'Shirt' 'Shorts' 'Shotgun' 'Shrimp'
'Sink' 'Skateboard' 'Ski' 'Skull' 'Skyscraper' 'Snail' 'Snake' 'Snowboard' 'Snowman' 'Sock'
'Sofa bed' 'Sombrero' 'Sparrow' 'Spider' 'Spoon' 'Sports uniform' 'Squirrel' 'Stairs'
'Starfish' 'Stool' 'Strawberry' 'Street light' 'Studio couch' 'Suit' 'Suitcase' 'Sun hat'
'Common sunflower' 'Sunglasses' 'Surfboard' 'Sushi' 'Swan' 'Swim cap' 'Swimming pool' 'Swimwear'
'Sword' 'Table tennis racket' 'Tablet computer' 'Taco' 'Tank' 'Tap' 'Tart' 'Taxi' 'Tea'
'Teapot' 'Teddy bear' 'Television' 'Tennis ball' 'Tennis racket' 'Tent' 'Tie' 'Tiger'
'Tin can' 'Tire' 'Toilet' 'Tomato' 'Tortoise' 'Tower' 'Traffic light' 'Train' 'Tripod'
'Trombone' 'Truck' 'Trumpet' 'Turkey' 'Umbrella' 'Van' 'Vase' 'Vehicle registration plate'
'Violin' 'Volleyball (Ball)' 'Waffle' 'Wall clock' 'Washing machine' 'Waste container' 'Watch'
'Watermelon' 'Whale' 'Wheel' 'Wheelchair' 'Whiteboard' 'Willow' 'Window' 'Window blind'
'Wine' 'Wine glass' 'Wok' 'Woman' 'Woodpecker' 'Zebra' 'Zucchini')

python fiftyone_openimages.py \
    --dataset-dir=${DATASET_PATH} \
    --output-labels="openimages-mlperf.json" \
    --classes "${MLPERF_CLASSES[@]}"
