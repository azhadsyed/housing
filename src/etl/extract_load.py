"""Extract Load.

Usage:
  extract_load.py -h | --help
  extract_load.py CITY CATEGORY [-tk]

Options:
  -h --help   Show this screen.
  -t --today  Download listings posted today (up until now)
  -k --keep   Store the downloaded listings to cache.json
"""

import json
from itertools import islice
from os import remove
from os.path import exists

import pymongo
from craigslist import CraigslistHousing
from docopt import docopt
from tqdm import tqdm

site = "newyork"
category = "apa"
today = False


# basic logic that fetches results from Craigslist
def extract(site, category, today=False):
    cl_h = CraigslistHousing(
        site=site,
        category=category,
        filters=dict(posted_today=today, has_image=True, bundle_duplicates=True),
    )
    results = cl_h.get_results(sort_by="newest", geotagged=True, include_details=True)
    results = [i for i in tqdm(results)]
    json.dump(results, "cache.json")


# now retrieve the existing Craigslist ID's from the raw data collection. init s/b 0
def load(documents):
    # on runtime, check if there's a cache
    if exists("cache.json"):
        with open("cache.json", "r") as f:
            documents = json.load(f)
    myclient = pymongo.MongoClient()
    db = myclient.housing
    listings = db.listings
    number_of_listings = listings.estimated_document_count()
    to_load = []
    if number_of_listings == 0:
        to_load = [listing for listing in documents]
    else:
        for listing in tqdm(documents):
            # check if the ID is in the db
            loading_id = listing["id"]
            query = {"id": loading_id}
            try:
                match = listings.find(query, {"_id": 0, "id": 1}).limit(1).next()
            # returns None if not present
            except StopIteration:
                match = None
            # add to the load queue if not already in db
            if not match:
                listings.insert(listing)
                to_load.append(listing)
    if to_load:
        job = listings.insert_many(to_load)
        print(f"{len(job.inserted_ids)} documents written to collection.")
    else:
        print("all of the generated listings were already present in the db.")
    # if the load completes, delete the cache so that next you check, no cache, and will commence pulling from CL


if __name__ == "__main__":
    arguments = docopt(__doc__, version="Extract Load 1.0")

    if not exists("cache.json"):
        extract(arguments["CITY"], arguments["CATEGORY"], arguments["--today"])

    documents = json.load("cache.json")
    load(documents)

    if not arguments["--keep"]:
        remove("cache.json")

# TODO: decouple database connection from load()