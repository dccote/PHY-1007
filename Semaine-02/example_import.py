from vectorfield import VectorField2D, SurfaceDomain

single_charge_field = VectorField2D() # Le domaine par défaut est -10 a 10 avec 19 points par dimension
single_charge_field.add_single_charge(xo=0, yo=0, q=1)
single_charge_field.display(use_color=True, title="Une seule charge")

two_charges_field = VectorField2D()
two_charges_field.add_single_charge(xo=5, yo=0, q=1)
two_charges_field.add_single_charge(xo=-5, yo=0, q=1)
two_charges_field.display(use_color=True, title="Deux charges positives")

dipole_field = VectorField2D()
dipole_field.add_single_charge(xo=5, yo=0, q=1)
dipole_field.add_single_charge(xo=-5, yo=0, q=-1)
dipole_field.display(use_color=True, title="Dipole")

quadrupole_field = VectorField2D()
quadrupole_field.add_single_charge(xo=5, yo=5, q=1)
quadrupole_field.add_single_charge(xo=-5, yo=5, q=-1)
quadrupole_field.add_single_charge(xo=5, yo=-5, q=-1)
quadrupole_field.add_single_charge(xo=-5, yo=-5, q=1)
quadrupole_field.display(use_color=True, title="Quadrupole")

