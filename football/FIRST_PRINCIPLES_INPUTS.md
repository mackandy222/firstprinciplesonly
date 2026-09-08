# Football Simulator - First Principles Inputs

## 1. Physics

- **Ball mass & radius** — determines inertia, air resistance, bounce
- **Ball coefficient of restitution** — how "bouncy" the ball is on impact
- **Ball drag coefficient** — air resistance on the ball in flight
- **Ball Magnus coefficient** — spin-induced curve (swerve/dip)
- **Ball spin vector (rpm + axis)** — applied at each kick
- **Gravity** — 9.81 m/s², affects trajectory arcs
- **Air density** — affects drag and Magnus force (varies with altitude/weather)
- **Ground friction coefficient** — rolling resistance on the pitch surface
- **Wind vector (speed + direction)** — external force on ball in flight

## 2. Pitch / Environment

- **Pitch dimensions** — length (100-110m) x width (64-75m)
- **Goal dimensions** — 7.32m x 2.44m
- **Pitch surface type** — grass, artificial turf (affects friction, bounce, player grip)
- **Pitch condition** — dry, wet, muddy, frozen (modifies friction coefficients)
- **Altitude** — affects air density, ball flight, player stamina drain
- **Temperature** — affects player fatigue rate
- **Weather state** — rain, snow, wind, clear (compound modifier)

## 3. Player - Physical Attributes

- **Max sprint speed (m/s)**
- **Acceleration (m/s²)**
- **Deceleration / braking rate**
- **Agility (direction change speed)**
- **Jump height / aerial reach**
- **Strength (mass + force in challenges)**
- **Height & weight** — collision physics, aerial duels
- **Stamina pool** — total energy available
- **Stamina drain rate** — energy cost per action (sprint, tackle, etc.)
- **Stamina recovery rate** — energy regain while walking/standing
- **Injury susceptibility** — probability modifier for injuries

## 4. Player - Technical Attributes

- **Kick power (max force applied to ball)**
- **Kick accuracy (angular error distribution)**
- **First touch quality** — ball control on reception (velocity damping)
- **Dribble close control** — ball displacement while running
- **Heading power & accuracy**
- **Tackling success probability**
- **Passing vision range** — how far ahead a player "sees" options
- **Weak foot ability** — penalty modifier for non-dominant foot
- **Curve/spin ability** — max spin a player can apply to the ball

## 5. Player - Mental / Decision Attributes

- **Positioning sense** — how well they find space or mark opponents
- **Decision speed (reaction time in ms)**
- **Composure under pressure** — accuracy modifier when pressed
- **Aggression** — likelihood to commit to challenges
- **Work rate** — willingness to run off-ball
- **Creativity** — probability of attempting risky/novel passes
- **Anticipation** — ability to read play and intercept
- **Leadership** — influence on nearby teammates' composure

## 6. Goalkeeper-Specific

- **Reflexes (reaction time)**
- **Dive reach (lateral + vertical)**
- **Handling (catch vs parry probability)**
- **Distribution accuracy (kicks & throws)**
- **Command of area (coming off the line)**
- **One-on-one ability** — shot-stopping in breakaways

## 7. Team Tactics / Formation

- **Formation shape** — positions of all 11 players (x,y coordinates)
- **Defensive line height** — how high/deep the backline sits
- **Pressing trigger distance** — when to close down the ball carrier
- **Pressing intensity** — how aggressively the team presses
- **Width in attack / defense**
- **Tempo** — target passing speed / directness
- **Build-up style** — short passing vs long ball probability
- **Offside trap usage**
- **Set piece routines** — corner/free-kick target positions
- **Marking type** — man-marking vs zonal

## 8. Player State (Runtime / Dynamic)

- **Current position (x, y)**
- **Current velocity vector**
- **Current stamina level**
- **Fatigue accumulation** — reduces all physical attributes over time
- **Morale / confidence** — dynamic modifier from match events
- **Card status** — yellow card increases caution
- **Injury status** — reduces physical output or forces substitution

## 9. Ball State (Runtime / Dynamic)

- **Position (x, y, z)**
- **Velocity vector (vx, vy, vz)**
- **Spin vector (wx, wy, wz)**
- **Possession state** — who controls it, or is it loose

## 10. Match Rules / Constants

- **Match duration** — 90 min (+ stoppage time)
- **Half-time break**
- **Substitution limit**
- **Offside rule geometry**
- **Foul probability model** — based on tackle type, speed, angle
- **Card thresholds** — foul severity -> yellow/red
- **VAR / referee accuracy** — probability of correct calls

## 11. Simulation Engine

- **Tick rate (dt)** — time step for physics updates (e.g., 60 Hz)
- **Collision detection model** — player-player, player-ball
- **Action resolution order** — who acts first on contested balls
- **Random seed** — for reproducibility

---

Everything above derives from three root primitives:
1. **Newtonian mechanics** — forces, masses, velocities, collisions
2. **Human biomechanics** — what a body can physically do
3. **Game rules** — the laws of football that constrain the simulation

i want to input when a real game begins what players pass to who and from this the settings of the players and stats must be creaed and hte tactics derived and as we get deeper in game est score and corners etc will start to be displayed 