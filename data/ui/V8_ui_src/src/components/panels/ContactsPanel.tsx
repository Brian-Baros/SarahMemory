import { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, Users, Plus, Search, Phone, Mail, MoreHorizontal, Loader2, X, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useSarahStore } from '@/stores/useSarahStore';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';

interface BackendContact {
  id: string;
  name: string;
  email?: string;
  phone?: string;
  address?: string;
  notes?: string;
  status?: 'online' | 'offline' | 'busy' | 'away';
}

export function ContactsPanel() {
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [backendContacts, setBackendContacts] = useState<BackendContact[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [editingContact, setEditingContact] = useState<BackendContact | null>(null);
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    address: '',
    notes: '',
  });

  const { contacts: localContacts, addContact: addLocalContact, deleteContact: deleteLocalContact } = useSarahStore();

  // Fetch contacts from backend
  const fetchContacts = async () => {
    setIsLoading(true);
    try {
      const response = await api.proxy.getContacts();
      if (response.contacts && Array.isArray(response.contacts)) {
        setBackendContacts(response.contacts as BackendContact[]);
      }
      setHasLoaded(true);
    } catch (error) {
      console.error('Failed to fetch contacts:', error);
      // Fall back to local contacts
      setHasLoaded(true);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-load on mount
  useEffect(() => {
    if (!hasLoaded) {
      fetchContacts();
    }
  }, [hasLoaded]);

  // Use backend contacts if available, otherwise local
  const contacts = backendContacts.length > 0 ? backendContacts : localContacts;

  const filteredContacts = contacts.filter(contact =>
    contact.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    contact.email?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    contact.phone?.includes(searchQuery)
  );

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'online': return 'bg-status-online';
      case 'away': return 'bg-status-warning';
      case 'busy': return 'bg-status-error';
      default: return 'bg-muted-foreground/50';
    }
  };

  const handleAddContact = async () => {
    if (!formData.name.trim()) {
      toast.error('Name is required');
      return;
    }

    try {
      // Try backend first
      await api.proxy.call('/api/contacts', {
        method: 'POST',
        body: formData,
      });
      toast.success('Contact added');
      fetchContacts(); // Refresh list
    } catch (error) {
      // Fall back to local storage
      addLocalContact({
        name: formData.name,
        email: formData.email || undefined,
        phone: formData.phone || undefined,
        address: formData.address || undefined,
        notes: formData.notes || undefined,
      });
      toast.success('Contact saved locally');
    }

    setFormData({ name: '', email: '', phone: '', address: '', notes: '' });
    setIsAddDialogOpen(false);
  };

  const handleEditContact = async () => {
    if (!editingContact) return;

    try {
      await api.proxy.call(`/api/contacts/${editingContact.id}`, {
        method: 'PUT',
        body: formData,
      });
      toast.success('Contact updated');
      fetchContacts();
    } catch (error) {
      toast.error('Failed to update contact');
    }

    setEditingContact(null);
    setFormData({ name: '', email: '', phone: '', address: '', notes: '' });
  };

  const handleDeleteContact = async (id: string) => {
    try {
      await api.proxy.call(`/api/contacts/${id}`, {
        method: 'DELETE',
      });
      toast.success('Contact deleted');
      fetchContacts();
    } catch (error) {
      // Try local delete
      deleteLocalContact(id);
      toast.success('Contact removed');
    }
  };

  const openEditDialog = (contact: BackendContact) => {
    setEditingContact(contact);
    setFormData({
      name: contact.name,
      email: contact.email || '',
      phone: contact.phone || '',
      address: contact.address || '',
      notes: contact.notes || '',
    });
  };

  return (
    <div className="p-3 space-y-3">
      {/* Search & Refresh */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search contacts..."
            className="pl-9 h-8 text-sm bg-sidebar-accent border-sidebar-border"
          />
        </div>
        <Button 
          variant="ghost" 
          size="icon" 
          className="h-8 w-8"
          onClick={fetchContacts}
          disabled={isLoading}
        >
          <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
        </Button>
      </div>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search contacts..."
                className="pl-9 h-8 text-sm bg-sidebar-accent border-sidebar-border"
              />
            </div>
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-8 w-8"
              onClick={fetchContacts}
              disabled={isLoading}
            >
              <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
            </Button>
          </div>

          {/* Contact List */}
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {isLoading && !hasLoaded ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : filteredContacts.length === 0 ? (
              <div className="text-center py-4 text-muted-foreground text-sm">
                No contacts found
              </div>
            ) : (
              filteredContacts.map((contact) => (
                <div
                  key={contact.id}
                  className="flex items-center justify-between p-2 rounded-lg bg-sidebar-accent/50 hover:bg-sidebar-accent transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                        <span className="text-xs font-medium">
                          {contact.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)}
                        </span>
                      </div>
                      <div className={cn(
                        "absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-2 border-sidebar",
                        getStatusColor(contact.status)
                      )} />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-foreground">{contact.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {contact.email || contact.phone || contact.address}
                      </div>
                    </div>
                  </div>

                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-7 w-7">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      {contact.phone && (
                        <DropdownMenuItem>
                          <Phone className="h-4 w-4 mr-2" /> Call
                        </DropdownMenuItem>
                      )}
                      {contact.email && (
                        <DropdownMenuItem>
                          <Mail className="h-4 w-4 mr-2" /> Email
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuItem onClick={() => openEditDialog(contact)}>
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuItem 
                        className="text-destructive"
                        onClick={() => handleDeleteContact(contact.id)}
                      >
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              ))
            )}
          </div>

          {/* Add Contact Button */}
          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm" className="w-full text-xs">
                <Plus className="h-3 w-3 mr-1.5" />
                Add Contact
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Add Contact</DialogTitle>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Name *</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="John Doe"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                    placeholder="john@example.com"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone">Phone</Label>
                  <Input
                    id="phone"
                    value={formData.phone}
                    onChange={(e) => setFormData(prev => ({ ...prev, phone: e.target.value }))}
                    placeholder="+1 555-0123"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="address">Address</Label>
                  <Input
                    id="address"
                    value={formData.address}
                    onChange={(e) => setFormData(prev => ({ ...prev, address: e.target.value }))}
                    placeholder="123 Main St"
                  />
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleAddContact}>
                    Add Contact
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>

          {/* Edit Contact Dialog */}
          <Dialog open={!!editingContact} onOpenChange={(open) => !open && setEditingContact(null)}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Edit Contact</DialogTitle>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <div className="space-y-2">
                  <Label htmlFor="edit-name">Name *</Label>
                  <Input
                    id="edit-name"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="edit-email">Email</Label>
                  <Input
                    id="edit-email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="edit-phone">Phone</Label>
                  <Input
                    id="edit-phone"
                    value={formData.phone}
                    onChange={(e) => setFormData(prev => ({ ...prev, phone: e.target.value }))}
                  />
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={() => setEditingContact(null)}>
                    Cancel
                  </Button>
                  <Button onClick={handleEditContact}>
                    Save Changes
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
    </div>
  );
}
