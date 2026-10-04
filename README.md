<div align="center">
  <img src="branding/logo.svg" alt="Lumina logo: a star, a passing planet, and the dip in starlight it causes" width="220"/>

# Lumina

**Volunteer computers searching NASA's public telescope data for new worlds.**

[Mission Control (live network)](https://mbarc.github.io/Lumina-Exoplanet-Hunter/) · [How it works](#how-it-works) · [Project status](#project-status)

</div>

---

## What Is This?

Lumina is an independent, open-source volunteer project in development, not affiliated with NASA. It uses spare computing time to search public records of how stars brighten and dim, looking for possible planets for astronomers to investigate. It runs in the background while you go about your day.

Participating computers form **ExoNet**: a volunteer-powered network working toward a single goal — finding worlds beyond our solar system.

---

## How Do You Find a Planet You Can't See?

When a planet passes in front of its star, it blocks a tiny fraction of the star's light — often less than one percent. A telescope watching that star records a brief, regular dip in brightness, repeating every time the planet completes an orbit. That is the picture in the Lumina logo: a star, a planet, and the dip it leaves in the line below.

Missions like Kepler watched hundreds of thousands of stars this way. Lumina's job is to look through those brightness records, star by star, for dips that repeat like a planet's would.

---

## The Problem

Space telescopes like Kepler, K2, and TESS have produced an enormous public archive of these brightness records. NASA makes them available so anyone can revisit the observations with new tools. Lumina aims to contribute by sharing the search across volunteer computers and flagging possible planet signals for further assessment.

---

## How It Works

This is the planned volunteer workflow once the installer is released:

1. **Install** the Lumina client on a Windows PC
2. It runs quietly in the **background** — no interaction required
3. It joins the ExoNet network and is handed a batch of stars nobody has checked yet
4. It downloads those stars' brightness records (Kepler today; TESS and K2 planned) and looks for repeating dips
5. Possible planet signals are sent back to the network for review

The more computers taking part, the faster the search goes.

---

## Project Status

Lumina is in active development by an independent volunteer.

| Mission | Status |
|---|---|
| **Kepler** | Being searched now — the detection model is trained and tested on Kepler data |
| **K2** | Planned |
| **TESS** (Transiting Exoplanet Survey Satellite) | Planned — next priority |

The Windows installer is not public yet. Signals Lumina flags are *candidates*: possible planets that need expert review and confirmation (by follow-up observation or statistical validation) before being accepted as planets.

---

## Getting Started

> *The installer is not public yet. Watch or star this repository to hear when it is.*

Once it's released, setup is meant to be: download the installer and run it. It configures your machine, sets up the background service, and connects you to the ExoNet network automatically. No astronomy or programming background required.

---

## Local Dashboard

Every ExoNet node includes a locally hosted web dashboard accessible from your browser. No account or internet connection required to view it — it runs entirely on your machine.

The dashboard lets you see:

- Which mission and sector your machine is currently processing
- How many stars have been analyzed and how many remain
- A live feed of light curves as they are processed, with transit detections highlighted
- Any candidate signals your machine has flagged for review
- Your node's contribution to the broader ExoNet network over time

It is designed to be left open in a browser tab — something you can glance at while working.

---

## If Your Machine Finds a Candidate

When your computer flags a possible transit signal, you will be notified through the dashboard. You will also have the opportunity to assign a **nickname** to the candidate — a name that will be associated with it permanently within ExoNet.

Nicknames are used within ExoNet only. Official names for exoplanets are chosen through separate **International Astronomical Union (IAU)** naming campaigns (NameExoWorlds); a discovery here does not guarantee eligibility or naming rights.

---

## For Researchers & Developers

Lumina is fully open source. If you are interested in contributing to the detection pipeline, extending mission support, or integrating candidate data into your own research workflows, see the project source and open an issue or pull request.

---

## Why This Matters

The data is already out there, paid for and made public. Every signal ExoNet flags is a star someone may want to take a closer look at. This project exists to put more eyes — and more computers — on that archive.

---

*Lumina is an independent open-source initiative and is not affiliated with NASA or any space agency.*
