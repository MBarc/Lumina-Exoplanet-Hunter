<div align="center">
  <img src="branding/logo.svg" alt="Lumina logo: a star, a passing planet, and the dip in starlight it causes" width="220"/>

# Lumina

Volunteer computers searching NASA's public telescope data for new worlds.

[Mission Control (live network)](https://mbarc.github.io/Lumina-Exoplanet-Hunter/) · [How it works](#how-it-works) · [Project status](#project-status)

</div>

## What this is

Lumina is an open-source volunteer project, still in development, that searches public telescope data for possible planets around other stars. It isn't affiliated with NASA. Once the client is released, it will run in the background on your computer while you work. All the computers running it together make up a network called ExoNet.

## How you find a planet you can't see

When a planet passes in front of its star, it blocks a small fraction of the star's light, often less than one percent. A telescope watching that star sees its brightness dip briefly, and the dip comes back every time the planet goes around. The Lumina logo shows exactly that: a star, a planet, and the dip in the line underneath.

Kepler watched hundreds of thousands of stars this way. NASA publishes those brightness records, along with the ones from K2 and TESS, so anyone can go back through them with new methods. Lumina goes through them star by star, looking for dips that repeat the way a planet's would, and passes anything promising to people for a closer look.

## How it works

This is how it will work once the installer is out:

1. You install the Lumina client on a Windows PC.
2. It runs in the background. You don't have to do anything.
3. ExoNet hands it a batch of stars from the queue.
4. It downloads those stars' brightness records (Kepler for now, with TESS and K2 planned) and looks for repeating dips.
5. It sends anything that looks like a planet back to the network for review.

The more computers join, the faster the search goes.

## Project status

One independent volunteer is building Lumina, and it's still in active development.

| Mission | Status |
|---|---|
| Kepler | Being searched now. The detection model is trained and tested on Kepler data. |
| K2 | Planned |
| TESS (Transiting Exoplanet Survey Satellite) | Planned, and next in line |

What Lumina flags are candidates, not planets. A candidate has to be reviewed by experts and confirmed, by follow-up observation or statistical validation, before it counts as a planet.

## Getting started

The Windows installer isn't public yet. Watch or star this repository to hear when it is.

The plan is for setup to be one step: download the installer and run it. It configures your machine, sets up the background service and connects to ExoNet on its own. You won't need any background in astronomy or programming.

## Local dashboard

Each computer running Lumina also serves a small web page you can open in your browser. It runs entirely on your machine, so you don't need an account or an internet connection to see it. It shows:

- which mission and sector your machine is working on
- how many stars it has analyzed and how many are left
- a live feed of light curves (brightness over time) as they're processed, with possible transits highlighted
- any candidates your machine has flagged
- how much your machine has added to ExoNet over time

It's meant to sit in a browser tab you can glance at while you work.

## If your machine finds a candidate

The dashboard will tell you, and you'll get to give the candidate a nickname. The nickname stays attached to it for good, but only inside ExoNet. Official exoplanet names come from separate naming campaigns run by the International Astronomical Union (NameExoWorlds), and a find here doesn't make anyone eligible for one.

## For researchers and developers

All of the code is open source. If you'd like to work on the detection pipeline, add another mission, or use the candidate data in your own research, open an issue or a pull request.

## Why it's worth doing

This data has already been collected and paid for, and it's public. Each signal ExoNet flags is a star someone may want to look at more closely. Lumina's job is to get more computers working through the archive.

*Lumina is an independent open-source project and is not affiliated with NASA or any other space agency.*
